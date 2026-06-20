// ============================================================
// benchmark.cpp — 性能基准测试
// 对比 Serial vs Parallel vs Hierarchical
// 所有 SlowXxx 继承自 Compute 节点（细粒度拆分后）
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
#include "nodes/BM25Node.h"
#include "nodes/FusionRerankerNode.h"
#include "nodes/ParentLookupNode.h"
#include "nodes/CrossEncoderNode.h"
#include "nodes/LLMGeneratorNode.h"

#include <chrono>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

static int g_delay_ms     = 50;
static int g_delay_jitter = 30;

// ---- 带随机延迟的子类 (继承自 Compute 节点) ----
class SlowEmbedComputeNode : public EmbedComputeNode {
public:
    CSTATUS run() override {
        CSTATUS st = EmbedComputeNode::run();
        int j = g_delay_jitter>0?(std::rand()%(g_delay_jitter*2))-g_delay_jitter:0;
        int d = g_delay_ms+j; if(d>0)std::this_thread::sleep_for(std::chrono::milliseconds(d));
        return st;
    }
};

class SlowQEmbedComputeNode : public QueryEmbedComputeNode {
public:
    CSTATUS run() override {
        CSTATUS st = QueryEmbedComputeNode::run();
        int j = g_delay_jitter>0?(std::rand()%(g_delay_jitter*2))-g_delay_jitter:0;
        int d = g_delay_ms+j; if(d>0)std::this_thread::sleep_for(std::chrono::milliseconds(d));
        return st;
    }
};

class SlowSearchComputeNode : public SearchComputeNode {
public:
    CSTATUS run() override {
        CSTATUS st = SearchComputeNode::run();
        int j = g_delay_jitter>0?(std::rand()%(g_delay_jitter*2))-g_delay_jitter:0;
        int d = g_delay_ms+j; if(d>0)std::this_thread::sleep_for(std::chrono::milliseconds(d));
        return st;
    }
};

class SlowBM25ComputeNode : public BM25ComputeNode {
public:
    CSTATUS run() override {
        CSTATUS st = BM25ComputeNode::run();
        int j = g_delay_jitter>0?(std::rand()%(g_delay_jitter*2))-g_delay_jitter:0;
        int d = g_delay_ms+j; if(d>0)std::this_thread::sleep_for(std::chrono::milliseconds(d));
        return st;
    }
};

class Timer {
    using C=std::chrono::high_resolution_clock; C::time_point t0_;
public:
    void start(){t0_=C::now();}
    double ms()const{return std::chrono::duration<double,std::milli>(C::now()-t0_).count();}
};

// ---- Serial ----
double run_serial(const std::vector<std::string>& files, const std::string& q){
    GPipelinePtr pl=GPipelineFactory::create();
    GElementPtr init,l,c,ec,em,s,qc,qm,shc,shm,pb,am;

    auto ebuf  = std::make_shared<EmbedIntermediate>();
    auto qbuf  = std::make_shared<QEmbedIntermediate>();
    auto sbuf  = std::make_shared<SearchIntermediate>();
    auto abuf  = std::make_shared<AnswerIntermediate>();

    pl->registerGElement<InitNode>(&init,{},"Init");
    pl->registerGElement<DocLoaderNode>(&l,{init},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<SlowEmbedComputeNode>(&ec,{c},"EC");
    pl->registerGElement<EmbedMergeNode>(&em,{ec},"EM");
    pl->registerGElement<QuerySetupNode>(&s,{em},"S");
    pl->registerGElement<SlowQEmbedComputeNode>(&qc,{s},"QC");
    pl->registerGElement<QueryEmbedMergeNode>(&qm,{qc},"QM");
    pl->registerGElement<SlowSearchComputeNode>(&shc,{qm},"SHC");
    pl->registerGElement<SearchMergeNode>(&shm,{shc},"SHM");
    pl->registerGElement<PromptBuildNode>(&pb,{shm},"PB");
    pl->registerGElement<AnswerMergeNode>(&am,{pb},"AM");

    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);
    dynamic_cast<EmbedComputeNode*>(ec)->setBuffer(ebuf);
    dynamic_cast<EmbedMergeNode*>(em)->setBuffer(ebuf);
    dynamic_cast<QueryEmbedComputeNode*>(qc)->setBuffer(qbuf);
    dynamic_cast<QueryEmbedMergeNode*>(qm)->setBuffer(qbuf);
    dynamic_cast<SearchComputeNode*>(shc)->setBuffer(sbuf);
    dynamic_cast<SearchMergeNode*>(shm)->setBuffer(sbuf);
    dynamic_cast<PromptBuildNode*>(pb)->setBuffer(abuf);
    dynamic_cast<AnswerMergeNode*>(am)->setBuffer(abuf);

    Timer t; t.start(); pl->process();
    double ms=t.ms(); GPipelineFactory::destroy(pl); return ms;
}

// ---- Parallel ----
double run_parallel(const std::vector<std::string>& files, const std::string& q){
    GPipelinePtr pl=GPipelineFactory::create();
    GElementPtr init,l,c,ec,em,s,dc;
    auto ebuf = std::make_shared<EmbedIntermediate>();

    pl->registerGElement<InitNode>(&init,{},"Init");
    pl->registerGElement<DocLoaderNode>(&l,{init},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<SlowEmbedComputeNode>(&ec,{c},"EC");
    pl->registerGElement<EmbedMergeNode>(&em,{ec},"EM");
    pl->registerGElement<QuerySetupNode>(&s,{em},"S");
    pl->registerGElement<QueryDecomposerNode>(&dc,{s},"DC");

    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);
    dynamic_cast<EmbedComputeNode*>(ec)->setBuffer(ebuf);
    dynamic_cast<EmbedMergeNode*>(em)->setBuffer(ebuf);

    GElementPtr qec[4],qem[4];
    std::shared_ptr<QEmbedIntermediate> qbuf[4];
    for(int i=0;i<4;++i){
        qbuf[i]=std::make_shared<QEmbedIntermediate>();
        char nm[16];snprintf(nm,sizeof(nm),"QEC%d",i);
        pl->registerGElement<SlowQEmbedComputeNode>(&qec[i],{dc},nm);
        dynamic_cast<QueryEmbedComputeNode*>(qec[i])->setSubQueryIndex(i);
        dynamic_cast<QueryEmbedComputeNode*>(qec[i])->setBuffer(qbuf[i]);
        snprintf(nm,sizeof(nm),"QEM%d",i);
        pl->registerGElement<QueryEmbedMergeNode>(&qem[i],{qec[i]},nm);
        dynamic_cast<QueryEmbedMergeNode*>(qem[i])->setBuffer(qbuf[i]);
    }

    GElementPtr shc[4][2],shm[4][2];
    std::shared_ptr<SearchIntermediate> sbuf[4][2];
    for(int i=0;i<4;++i)for(int j=0;j<2;++j){
        sbuf[i][j]=std::make_shared<SearchIntermediate>();
        char nm[16];snprintf(nm,sizeof(nm),"SHC%d%c",i,'A'+j);
        pl->registerGElement<SlowSearchComputeNode>(&shc[i][j],{qem[i]},nm);
        dynamic_cast<SearchComputeNode*>(shc[i][j])->configure(10,j,2,i);
        dynamic_cast<SearchComputeNode*>(shc[i][j])->setBuffer(sbuf[i][j]);
        snprintf(nm,sizeof(nm),"SHM%d%c",i,'A'+j);
        pl->registerGElement<SearchMergeNode>(&shm[i][j],{shc[i][j]},nm);
        dynamic_cast<SearchMergeNode*>(shm[i][j])->setBuffer(sbuf[i][j]);
    }

    GElementPtr fc01,fc23,fm01,fm23,pb,am;
    auto fbuf01=std::make_shared<FusionIntermediate>();
    auto fbuf23=std::make_shared<FusionIntermediate>();
    auto abuf=std::make_shared<AnswerIntermediate>();

    pl->registerGElement<FusionComputeNode>(&fc01,{shm[0][0],shm[0][1],shm[1][0],shm[1][1]},"FC01");
    pl->registerGElement<FusionComputeNode>(&fc23,{shm[2][0],shm[2][1],shm[3][0],shm[3][1]},"FC23");
    pl->registerGElement<FusionMergeNode>(&fm01,{fc01},"FM01");
    pl->registerGElement<FusionMergeNode>(&fm23,{fc23},"FM23");
    pl->registerGElement<PromptBuildNode>(&pb,{fm01,fm23},"PB");
    pl->registerGElement<AnswerMergeNode>(&am,{pb},"AM");

    dynamic_cast<FusionComputeNode*>(fc01)->setBuffer(fbuf01);
    dynamic_cast<FusionComputeNode*>(fc23)->setBuffer(fbuf23);
    dynamic_cast<FusionMergeNode*>(fm01)->setBuffer(fbuf01);
    dynamic_cast<FusionMergeNode*>(fm23)->setBuffer(fbuf23);
    dynamic_cast<PromptBuildNode*>(pb)->setBuffer(abuf);
    dynamic_cast<AnswerMergeNode*>(am)->setBuffer(abuf);

    Timer t; t.start(); pl->process();
    double ms=t.ms(); GPipelineFactory::destroy(pl); return ms;
}

// ---- Hierarchical ----
double run_hierarchical(const std::vector<std::string>& files, const std::string& q){
    GPipelinePtr pl=GPipelineFactory::create();
    auto ebuf=std::make_shared<EmbedIntermediate>();

    GElementPtr init,l,c,ec,em,s,dc;
    pl->registerGElement<InitNode>(&init,{},"Init");
    pl->registerGElement<DocLoaderNode>(&l,{init},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<SlowEmbedComputeNode>(&ec,{c},"EC");
    pl->registerGElement<EmbedMergeNode>(&em,{ec},"EM");
    pl->registerGElement<QuerySetupNode>(&s,{em},"S");
    pl->registerGElement<QueryDecomposerNode>(&dc,{s},"DC");

    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);
    dynamic_cast<EmbedComputeNode*>(ec)->setBuffer(ebuf);
    dynamic_cast<EmbedMergeNode*>(em)->setBuffer(ebuf);

    GElementPtr qec[4],qem[4];
    std::shared_ptr<QEmbedIntermediate> qbuf[4];
    for(int i=0;i<4;++i){
        qbuf[i]=std::make_shared<QEmbedIntermediate>();
        char nm[16];snprintf(nm,sizeof(nm),"QEC%d",i);
        pl->registerGElement<SlowQEmbedComputeNode>(&qec[i],{dc},nm);
        dynamic_cast<QueryEmbedComputeNode*>(qec[i])->setSubQueryIndex(i);
        dynamic_cast<QueryEmbedComputeNode*>(qec[i])->setBuffer(qbuf[i]);
        snprintf(nm,sizeof(nm),"QEM%d",i);
        pl->registerGElement<QueryEmbedMergeNode>(&qem[i],{qec[i]},nm);
        dynamic_cast<QueryEmbedMergeNode*>(qem[i])->setBuffer(qbuf[i]);
    }

    GElementPtr shc[4][2],shm[4][2];
    std::shared_ptr<SearchIntermediate> sbuf[4][2];
    for(int i=0;i<4;++i)for(int j=0;j<2;++j){
        sbuf[i][j]=std::make_shared<SearchIntermediate>();
        char nm[16];snprintf(nm,sizeof(nm),"SHC%d%c",i,'A'+j);
        pl->registerGElement<SlowSearchComputeNode>(&shc[i][j],{qem[i]},nm);
        dynamic_cast<SearchComputeNode*>(shc[i][j])->configure(25,j,2,i);
        dynamic_cast<SearchComputeNode*>(shc[i][j])->setBuffer(sbuf[i][j]);
        snprintf(nm,sizeof(nm),"SHM%d%c",i,'A'+j);
        pl->registerGElement<SearchMergeNode>(&shm[i][j],{shc[i][j]},nm);
        dynamic_cast<SearchMergeNode*>(shm[i][j])->setBuffer(sbuf[i][j]);
    }

    GElementPtr bmc,bmm;
    auto bmbuf=std::make_shared<BM25Intermediate>();
    pl->registerGElement<SlowBM25ComputeNode>(&bmc,{em},"BMC");
    pl->registerGElement<BM25MergeNode>(&bmm,{bmc},"BMM");
    dynamic_cast<BM25ComputeNode*>(bmc)->configure(50);
    dynamic_cast<BM25ComputeNode*>(bmc)->setBuffer(bmbuf);
    dynamic_cast<BM25MergeNode*>(bmm)->setBuffer(bmbuf);

    GElementPtr fc,fm,pc,pm,cec[2],cem[2],pb,am;
    auto fbuf=std::make_shared<FusionIntermediate>();
    auto pbuf=std::make_shared<ParentIntermediate>();
    std::shared_ptr<CEIntermediate> cebuf[2]{
        std::make_shared<CEIntermediate>(),
        std::make_shared<CEIntermediate>()};
    auto abuf=std::make_shared<AnswerIntermediate>();

    std::set<GElement*> fusion_deps={shm[0][0],shm[0][1],shm[1][0],shm[1][1],
                                     shm[2][0],shm[2][1],shm[3][0],shm[3][1],bmm};
    pl->registerGElement<FusionComputeNode>(&fc,fusion_deps,"FC");
    pl->registerGElement<FusionMergeNode>(&fm,{fc},"FM");
    pl->registerGElement<ParentComputeNode>(&pc,{fm},"PC");
    pl->registerGElement<ParentMergeNode>(&pm,{pc},"PM");

    for(int i=0;i<2;++i){
        char nm[16];
        snprintf(nm,sizeof(nm),"CEC%d",i+1);
        pl->registerGElement<CEComputeNode>(&cec[i],{pm},nm);
        dynamic_cast<CEComputeNode*>(cec[i])->setBuffer(cebuf[i]);
        snprintf(nm,sizeof(nm),"CEM%d",i+1);
        pl->registerGElement<CEMergeNode>(&cem[i],{cec[i]},nm);
        dynamic_cast<CEMergeNode*>(cem[i])->setBuffer(cebuf[i]);
    }

    pl->registerGElement<PromptBuildNode>(&pb,{cem[0],cem[1]},"PB");
    pl->registerGElement<AnswerMergeNode>(&am,{pb},"AM");

    dynamic_cast<FusionComputeNode*>(fc)->setBuffer(fbuf);
    dynamic_cast<FusionMergeNode*>(fm)->setBuffer(fbuf);
    dynamic_cast<ParentComputeNode*>(pc)->setBuffer(pbuf);
    dynamic_cast<ParentMergeNode*>(pm)->setBuffer(pbuf);
    dynamic_cast<PromptBuildNode*>(pb)->setBuffer(abuf);
    dynamic_cast<AnswerMergeNode*>(am)->setBuffer(abuf);

    Timer t; t.start(); pl->process();
    double ms=t.ms(); GPipelineFactory::destroy(pl); return ms;
}

void print_batch(const char* l, std::vector<double>& d){
    double s=0,lo=1e9,hi=0;for(double v:d){s+=v;if(v<lo)lo=v;if(v>hi)hi=v;}
    double av=s/d.size(),vr=0;for(double v:d)vr+=(v-av)*(v-av);
    CGRAPH_ECHO("  %-14s avg=%7.1f ms  min=%6.1f  max=%6.1f  std=%.1f",
                l,av,lo,hi,std::sqrt(vr/d.size()));
}

int main(int argc, char* argv[]){
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::srand((unsigned)std::chrono::system_clock::now().time_since_epoch().count());

    int runs=5;
    std::vector<std::string> files;
    std::string q="Compare C++11 and C++17 concurrency features and performance?";

    for(int i=1;i<argc;++i){
        std::string a(argv[i]);
        if(a.rfind("--delay=",0)==0)      g_delay_ms=std::stoi(a.substr(8));
        else if(a.rfind("--jitter=",0)==0) g_delay_jitter=std::stoi(a.substr(9));
        else if(a.rfind("--runs=",0)==0)  runs=std::stoi(a.substr(7));
        else if(a.rfind("--query=",0)==0) q=a.substr(8);
        else files.push_back(a);
    }
    if(files.empty()) files.push_back("test_docs/cpp_concurrency_full.txt");

    CGRAPH_ECHO("========== BENCHMARK delay=%dms jitter=%dms runs=%d ==========",
                g_delay_ms,g_delay_jitter,runs);

    // Warm-up
    run_serial(files,q); run_parallel(files,q); run_hierarchical(files,q);

    std::vector<double> st(runs),pt(runs),ht(runs);

    CGRAPH_ECHO("--- SERIAL (E=2 S=1) ---");
    for(int i=0;i<runs;++i){st[i]=run_serial(files,q);CGRAPH_ECHO("  r%d:%.1f ms",i+1,st[i]);}

    CGRAPH_ECHO("--- PARALLEL (E=5 S=8 R=2) ---");
    for(int i=0;i<runs;++i){pt[i]=run_parallel(files,q);CGRAPH_ECHO("  r%d:%.1f ms",i+1,pt[i]);}

    CGRAPH_ECHO("--- HIERARCHICAL (E=5 S=8 BM25=1 PL=1 CE=2) ---");
    for(int i=0;i<runs;++i){ht[i]=run_hierarchical(files,q);CGRAPH_ECHO("  r%d:%.1f ms",i+1,ht[i]);}

    print_batch("Serial",st); print_batch("Parallel",pt); print_batch("Hierarchical",ht);

    double sa=0,pa=0,ha=0;
    for(double v:st)sa+=v; for(double v:pt)pa+=v; for(double v:ht)ha+=v;
    sa/=runs; pa/=runs; ha/=runs;

    CGRAPH_ECHO("========== RESULT ==========");
    CGRAPH_ECHO("Serial        avg: %.1f ms  (E=2  S=1)", sa);
    CGRAPH_ECHO("Parallel      avg: %.1f ms  (E=5  S=8  R=2)", pa);
    CGRAPH_ECHO("Hierarchical  avg: %.1f ms  (E=5  S=8  BM25=1  PL=1  CE=2)", ha);
    CGRAPH_ECHO("Parallel    vs Serial: %+.1f%%  speedup=%.2fx", (pa-sa)/sa*100.0, sa/pa);
    CGRAPH_ECHO("Hierarch    vs Serial: %+.1f%%  speedup=%.2fx", (ha-sa)/sa*100.0, sa/ha);
    CGRAPH_ECHO("Work: Serial=3ops  Parallel=13ops  Hierarchical=17+ops at similar latency");
}
