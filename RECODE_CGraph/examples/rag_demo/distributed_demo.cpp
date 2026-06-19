// ============================================================
// distributed_demo.cpp — 分布式 RAG 架构单进程仿真
//
// 模拟: 3 Retriever + 1 Generator A2A 分布式部署
// 内部: CGraph DAG 并行调度所有 A2A 调用
//
// DAG: Decomposer→3×QE→3×A2A_Retriever(并行)→Fusion→A2A_Generator→答案
// ============================================================

#include "../../src/GraphCtrl/GraphInclude.h"
#include "RAGCommon.h"
#include "nodes/DocLoaderNode.h"
#include "nodes/ChunkerNode.h"
#include "nodes/EmbedderNode.h"
#include <EmbeddingClient.h>
#include "nodes/QuerySetupNode.h"
#include "nodes/QueryDecomposerNode.h"
#include "nodes/VectorSearchNode.h"
#include "nodes/FusionRerankerNode.h"
#include "nodes/LLMGeneratorNode.h"

#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

// ---- 模拟 Retriever Agent ----
struct RetrieverSim {
    std::string name;
    RetrieverSim(const std::string& n) : name(n) {}
    std::vector<std::pair<int,float>> retrieve(
        const std::vector<std::vector<float>>& embs,
        const std::vector<float>& qv, int top_k)
    {
        std::vector<std::pair<int,float>> r;
        for(size_t i=0;i<embs.size();++i)
            r.push_back({(int)i, cosine_similarity(qv, embs[i])});
        std::sort(r.begin(),r.end(),
            [](auto& a,auto& b){return a.second>b.second;});
        if((int)r.size()>top_k) r.resize(top_k);
        CGRAPH_ECHO("[RAG]   A2A->Retriever[%s]: top1=%.4f",
                    name.c_str(), r.empty()?0.0f:r[0].second);
        return r;
    }
};

// ---- A2A Retriever Call Node ----
class A2ARetrieverCallNode : public GNode {
    RetrieverSim* target_ = nullptr;
    int query_vec_idx_ = 0;
    int top_k_ = 5;
public:
    A2ARetrieverCallNode() { this->setName("A2A_Retriever"); }
    void bind(RetrieverSim* t, int qvi, int k=5) {
        target_=t; query_vec_idx_=qvi; top_k_=k;
        this->setName("A2A_"+t->name);
    }
    CSTATUS run() override {
        auto* p=this->getGParam<RAGParam>("rag");
        if(!p||!target_) return STATUS_ERR;
        std::vector<float> qv;
        {CGRAPH_PARAM_READ_REGION(p){
            if(query_vec_idx_>=(int)p->query_embeddings.size())return STATUS_ERR;
            qv=p->query_embeddings[query_vec_idx_];}}
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        auto res=target_->retrieve(p->embeddings,qv,top_k_);
        CGRAPH_PARAM_WRITE_REGION(p){p->top_k.push_back(res);}
        return STATUS_OK;
    }
};

// ---- A2A Generator Call Node ----
class A2AGeneratorCallNode : public GNode {
public:
    A2AGeneratorCallNode() { this->setName("A2A_Generator"); }
    CSTATUS run() override {
        auto* p=this->getGParam<RAGParam>("rag");
        if(!p) return STATUS_ERR;
        std::ostringstream ctx; std::string q;
        {CGRAPH_PARAM_READ_REGION(p){
            for(auto& kv:p->fused_results)
                if(kv.first>=0&&kv.first<(int)p->chunks.size())
                    ctx<<p->chunks[kv.first]<<"\n---\n";
            q=p->query;}}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::string ans="[Distributed RAG Answer]\nQuery: "+q
            +"\nContexts: "+std::to_string(p->fused_results.size())
            +"\n---\n"+ctx.str().substr(0,400);
        CGRAPH_PARAM_WRITE_REGION(p){p->answer=ans;}
        CGRAPH_ECHO("[RAG]   A2A->Generator: %zu chars", ans.size());
        return STATUS_OK;
    }
};

// ================================================================
int main(int argc, char* argv[]){
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    EmbedderNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());
    std::string q; std::vector<std::string> files;
    for(int i=1;i<argc;++i){
        std::string a(argv[i]);
        if(a.rfind("--query=",0)==0) q=a.substr(8); else files.push_back(a);
    }
    if(files.empty()) files.push_back("test_docs/cpp_concurrency_full.txt");
    if(q.empty()) q="Compare C++11 and C++17 concurrency?";

    CGRAPH_ECHO("[RAG] ====================================");
    CGRAPH_ECHO("[RAG] DISTRIBUTED RAG (single-process sim)");
    CGRAPH_ECHO("[RAG] ====================================");
    CGRAPH_ECHO("[RAG] Architecture: Orchestrator internal CGraph DAG");
    CGRAPH_ECHO("[RAG]   3 x A2A Retriever calls (parallel)");
    CGRAPH_ECHO("[RAG]   1 x A2A Generator  call");
    CGRAPH_ECHO("[RAG] Query: %s", q.c_str());

    GPipelinePtr pl=GPipelineFactory::create();

    // Layer 1: Build + Decompose
    GElementPtr l,c,de,s,dc;
    pl->registerGElement<DocLoaderNode>(&l,{},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<EmbedderNode>(&de,{c},"DE");
    pl->registerGElement<QuerySetupNode>(&s,{de},"S");
    pl->registerGElement<QueryDecomposerNode>(&dc,{s},"DC");
    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);

    // Layer 2: 3 query embeddings
    GElementPtr qe[3];
    for(int i=0;i<3;++i){
        char nm[16];snprintf(nm,16,"QE%d",i);
        pl->registerGElement<EmbedderNode>(&qe[i],{dc},nm);
        dynamic_cast<EmbedderNode*>(qe[i])->setQueryMode(true);
        dynamic_cast<EmbedderNode*>(qe[i])->setSubQueryIndex(i);
    }

    // Layer 3: 3 parallel A2A calls
    RetrieverSim r1("R1"),r2("R2"),r3("R3");
    GElementPtr cr1,cr2,cr3;
    pl->registerGElement<A2ARetrieverCallNode>(&cr1,{qe[0]},"CR1");
    pl->registerGElement<A2ARetrieverCallNode>(&cr2,{qe[1]},"CR2");
    pl->registerGElement<A2ARetrieverCallNode>(&cr3,{qe[2]},"CR3");
    dynamic_cast<A2ARetrieverCallNode*>(cr1)->bind(&r1,0);
    dynamic_cast<A2ARetrieverCallNode*>(cr2)->bind(&r2,1);
    dynamic_cast<A2ARetrieverCallNode*>(cr3)->bind(&r3,2);

    // Layer 4: Fusion
    GElementPtr fusion;
    pl->registerGElement<FusionRerankerNode>(&fusion,{cr1,cr2,cr3},"Fusion");

    // Layer 5: A2A Generator call
    GElementPtr cg;
    pl->registerGElement<A2AGeneratorCallNode>(&cg,{fusion},"CG");

    CGRAPH_ECHO("[RAG] DAG ready. Running...");
    pl->process();

    CGRAPH_ECHO("[RAG] ====================================");
    CGRAPH_ECHO("[RAG] Distributed RAG completed.");
    CGRAPH_ECHO("[RAG] ====================================");

    GPipelineFactory::destroy(pl);
    return 0;
}
