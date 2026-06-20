// ============================================================
// benchmark.cpp — 性能基准测试
// Phase 6 扩展: 增加 Hierarchical 模式对比
// 对比 Serial vs Parallel vs Hierarchical
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

// ---- 带随机延迟的子类 ----
class SlowEmbedderNode : public EmbedderNode {
public:
    CSTATUS run() override {
        CSTATUS st = EmbedderNode::run();
        int j = g_delay_jitter>0?(std::rand()%(g_delay_jitter*2))-g_delay_jitter:0;
        int d = g_delay_ms+j; if(d>0)std::this_thread::sleep_for(std::chrono::milliseconds(d));
        return st;
    }
};
class SlowSearchNode : public VectorSearchNode {
public:
    CSTATUS run() override {
        CSTATUS st = VectorSearchNode::run();
        int j = g_delay_jitter>0?(std::rand()%(g_delay_jitter*2))-g_delay_jitter:0;
        int d = g_delay_ms+j; if(d>0)std::this_thread::sleep_for(std::chrono::milliseconds(d));
        return st;
    }
};
class SlowBM25Node : public BM25Node {
public:
    CSTATUS run() override {
        CSTATUS st = BM25Node::run();
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
    GElementPtr l,c,de,s,qe,sh,g;
    pl->registerGElement<DocLoaderNode>(&l,{},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<SlowEmbedderNode>(&de,{c},"DE");
    pl->registerGElement<QuerySetupNode>(&s,{de},"S");
    pl->registerGElement<SlowEmbedderNode>(&qe,{s},"QE");
    pl->registerGElement<SlowSearchNode>(&sh,{qe},"SH");
    pl->registerGElement<LLMGeneratorNode>(&g,{sh},"G");
    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);
    dynamic_cast<EmbedderNode*>(qe)->setQueryMode(true);
    Timer t; t.start(); pl->process();
    double ms=t.ms(); GPipelineFactory::destroy(pl); return ms;
}

// ---- Parallel ----
double run_parallel(const std::vector<std::string>& files, const std::string& q){
    GPipelinePtr pl=GPipelineFactory::create();
    GElementPtr l,c,de,s,dc;
    pl->registerGElement<DocLoaderNode>(&l,{},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<SlowEmbedderNode>(&de,{c},"DE");
    pl->registerGElement<QuerySetupNode>(&s,{de},"S");
    pl->registerGElement<QueryDecomposerNode>(&dc,{s},"DC");
    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);

    GElementPtr qe[4],sh[4][2];
    for(int i=0;i<4;++i){
        char nm[16];snprintf(nm,sizeof(nm),"QE%d",i);
        pl->registerGElement<SlowEmbedderNode>(&qe[i],{dc},nm);
        dynamic_cast<EmbedderNode*>(qe[i])->setQueryMode(true);
        dynamic_cast<EmbedderNode*>(qe[i])->setSubQueryIndex(i);
    }
    for(int i=0;i<4;++i)for(int j=0;j<2;++j){
        char nm[16];snprintf(nm,sizeof(nm),"S%d%c",i,'A'+j);
        pl->registerGElement<SlowSearchNode>(&sh[i][j],{qe[i]},nm);
        dynamic_cast<VectorSearchNode*>(sh[i][j])->configure(10,j,2,i);
    }
    GElementPtr r01,r23,g;
    pl->registerGElement<FusionRerankerNode>(&r01,{sh[0][0],sh[0][1],sh[1][0],sh[1][1]},"R01");
    pl->registerGElement<FusionRerankerNode>(&r23,{sh[2][0],sh[2][1],sh[3][0],sh[3][1]},"R23");
    pl->registerGElement<LLMGeneratorNode>(&g,{r01,r23},"G");

    Timer t; t.start(); pl->process();
    double ms=t.ms(); GPipelineFactory::destroy(pl); return ms;
}

// ---- Hierarchical (Phase 6) ----
double run_hierarchical(const std::vector<std::string>& files, const std::string& q){
    GPipelinePtr pl=GPipelineFactory::create();
    GElementPtr l,c,de,s,dc;
    pl->registerGElement<DocLoaderNode>(&l,{},"L");
    pl->registerGElement<ChunkerNode>(&c,{l},"C");
    pl->registerGElement<SlowEmbedderNode>(&de,{c},"DE");
    pl->registerGElement<QuerySetupNode>(&s,{de},"S");
    pl->registerGElement<QueryDecomposerNode>(&dc,{s},"DC");
    dynamic_cast<DocLoaderNode*>(l)->setPaths(files);
    dynamic_cast<QuerySetupNode*>(s)->setQuery(q);

    GElementPtr qe[4];
    for(int i=0;i<4;++i){
        char nm[16];snprintf(nm,sizeof(nm),"QE%d",i);
        pl->registerGElement<SlowEmbedderNode>(&qe[i],{dc},nm);
        dynamic_cast<EmbedderNode*>(qe[i])->setQueryMode(true);
        dynamic_cast<EmbedderNode*>(qe[i])->setSubQueryIndex(i);
    }
    GElementPtr sh[4][2];
    for(int i=0;i<4;++i)for(int j=0;j<2;++j){
        char nm[16];snprintf(nm,sizeof(nm),"S%d%c",i,'A'+j);
        pl->registerGElement<SlowSearchNode>(&sh[i][j],{qe[i]},nm);
        dynamic_cast<VectorSearchNode*>(sh[i][j])->configure(25,j,2,i);
    }
    GElementPtr bm25,plk,ce1,ce2,g;
    pl->registerGElement<SlowBM25Node>(&bm25,{de},"BM25");
    dynamic_cast<BM25Node*>(bm25)->configure(50);

    GElementPtr fusion;
    std::set<GElement*> fusion_deps={sh[0][0],sh[0][1],sh[1][0],sh[1][1],sh[2][0],sh[2][1],sh[3][0],sh[3][1],bm25};
    pl->registerGElement<FusionRerankerNode>(&fusion,fusion_deps,"FusionHybrid");
    pl->registerGElement<ParentLookupNode>(&plk,{fusion},"ParentLookup");
    pl->registerGElement<CrossEncoderNode>(&ce1,{plk},"CE1");
    pl->registerGElement<CrossEncoderNode>(&ce2,{plk},"CE2");
    pl->registerGElement<LLMGeneratorNode>(&g,{ce1,ce2},"G");

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

    // EmbedderNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

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
