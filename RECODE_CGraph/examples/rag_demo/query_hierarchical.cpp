// ============================================================
// query_hierarchical.cpp 鈥?灞傛绱㈠紩 + 娣峰悎妫€绱?+ 澶氱骇婕忔枟 RAG
//
// Phase 6 鏍稿績 DAG锛堢害 27 鑺傜偣, 6 灞傚苟琛岀獥鍙ｏ級:
//
// Layer 1: Loader 鈫?Chunker(dual-output) 鈫?DocEmbed (涓茶寤哄簱)
// Layer 2: Setup 鈫?Decomposer (鏌ヨ鍒濆鍖?
// Layer 3: 4脳QE (骞惰 query embedding)
// Layer 4: 8脳DenseSearch + 1脳BM25 (9璺苟琛屾绱?
// Layer 5: FusionReranker (BM25+Dense 鍔犳潈铻嶅悎 鈫?Top-50)
// Layer 6: ParentLookup 鈫?2脳CrossEncoder 鈫?Generator
//          (灞傛绱㈠紩 + 绮炬帓 + 鐢熸垚)
//
// 鍏辩害 27 鑺傜偣锛? 灞傚苟琛岀獥鍙ｏ紝灞曠ず瀹屾暣鐨?CGraph DAG 璋冨害鑳藉姏
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

#include <string>
#include <vector>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string question;
    std::vector<std::string> file_paths;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg.rfind("--query=", 0) == 0)
                question = arg.substr(8);
            else
                file_paths.push_back(arg);
        }
    }
    if (file_paths.empty())
        file_paths.push_back("test_docs/cpp_concurrency_full.txt");
    if (question.empty())
        question = "What are the key differences between C++11 and C++17 concurrency?";

    CGRAPH_ECHO("[RAG] ============================================");
    CGRAPH_ECHO("[RAG] HIERARCHICAL RAG (Phase 6)");
    CGRAPH_ECHO("[RAG] Hierarchy: Small chunks (index) -> Parent paragraphs (LLM)");
    CGRAPH_ECHO("[RAG] Hybrid: BM25 + Dense alpha=0.7");
    CGRAPH_ECHO("[RAG] Funnel: Recall 200 -> Filter 50 -> Rerank 5");
    CGRAPH_ECHO("[RAG] %zu file(s), Query: %s",
                file_paths.size(), question.c_str());
    CGRAPH_ECHO("[RAG] ============================================");

    // 鏃?API key 鏃惰烦杩?EmbeddingClient 鍒涘缓锛岀洿鎺ヤ娇鐢?mock锛堥伩鍏?curl_global_init 闃诲锛?    // 鏈?key 鏃跺彇娑堟敞閲婁笅涓€琛? EmbedderNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    auto t_start = std::chrono::high_resolution_clock::now();

    GPipelinePtr pipeline = GPipelineFactory::create();

    // ====== Layer 1: 寤哄簱锛? 鑺傜偣, 涓茶锛?======
    GElementPtr init, loader, chunker, doc_embed;
    pipeline->registerGElement<InitNode>(&init, {}, "Init"); pipeline->registerGElement<DocLoaderNode>(&loader, {init}, "DocLoader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedderNode>(&doc_embed, {chunker}, "DocEmbedder");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);

    // ====== Layer 2: 鏌ヨ鍒濆鍖栵紙2 鑺傜偣, 涓茶锛?======
    GElementPtr setup, decomposer;
    pipeline->registerGElement<QuerySetupNode>(&setup, {doc_embed}, "QuerySetup");
    pipeline->registerGElement<QueryDecomposerNode>(&decomposer, {setup}, "Decomposer");

    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);

    // ====== Layer 3: 4璺苟琛?Embedding ======
    GElementPtr qe0, qe1, qe2, qe3;
    pipeline->registerGElement<EmbedderNode>(&qe0, {decomposer}, "QE_SQ0");
    pipeline->registerGElement<EmbedderNode>(&qe1, {decomposer}, "QE_SQ1");
    pipeline->registerGElement<EmbedderNode>(&qe2, {decomposer}, "QE_SQ2");
    pipeline->registerGElement<EmbedderNode>(&qe3, {decomposer}, "QE_SQ3");

    dynamic_cast<EmbedderNode*>(qe0)->setQueryMode(true);
    dynamic_cast<EmbedderNode*>(qe0)->setSubQueryIndex(0);
    dynamic_cast<EmbedderNode*>(qe1)->setQueryMode(true);
    dynamic_cast<EmbedderNode*>(qe1)->setSubQueryIndex(1);
    dynamic_cast<EmbedderNode*>(qe2)->setQueryMode(true);
    dynamic_cast<EmbedderNode*>(qe2)->setSubQueryIndex(2);
    dynamic_cast<EmbedderNode*>(qe3)->setQueryMode(true);
    dynamic_cast<EmbedderNode*>(qe3)->setSubQueryIndex(3);

    // ====== Layer 4: 9璺苟琛屾绱?(8 Dense + 1 BM25) ======
    GElementPtr s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b;
    pipeline->registerGElement<VectorSearchNode>(&s0a, {qe0}, "S0A");
    pipeline->registerGElement<VectorSearchNode>(&s0b, {qe0}, "S0B");
    pipeline->registerGElement<VectorSearchNode>(&s1a, {qe1}, "S1A");
    pipeline->registerGElement<VectorSearchNode>(&s1b, {qe1}, "S1B");
    pipeline->registerGElement<VectorSearchNode>(&s2a, {qe2}, "S2A");
    pipeline->registerGElement<VectorSearchNode>(&s2b, {qe2}, "S2B");
    pipeline->registerGElement<VectorSearchNode>(&s3a, {qe3}, "S3A");
    pipeline->registerGElement<VectorSearchNode>(&s3b, {qe3}, "S3B");

    dynamic_cast<VectorSearchNode*>(s0a)->configure(25, 0, 2, 0);
    dynamic_cast<VectorSearchNode*>(s0b)->configure(25, 1, 2, 0);
    dynamic_cast<VectorSearchNode*>(s1a)->configure(25, 0, 2, 1);
    dynamic_cast<VectorSearchNode*>(s1b)->configure(25, 1, 2, 1);
    dynamic_cast<VectorSearchNode*>(s2a)->configure(25, 0, 2, 2);
    dynamic_cast<VectorSearchNode*>(s2b)->configure(25, 1, 2, 2);
    dynamic_cast<VectorSearchNode*>(s3a)->configure(25, 0, 2, 3);
    dynamic_cast<VectorSearchNode*>(s3b)->configure(25, 1, 2, 3);

    // 1 BM25 node
    GElementPtr bm25;
    pipeline->registerGElement<BM25Node>(&bm25, {doc_embed}, "BM25");
    dynamic_cast<BM25Node*>(bm25)->configure(50);

    // ====== Layer 5: 娣峰悎铻嶅悎 (Dense + BM25 -> Top-50) ======
    GElementPtr fusion;
    pipeline->registerGElement<FusionRerankerNode>(&fusion,
        {s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b, bm25},
        "FusionHybrid");
    // Note: configure() is called via constructor-like approach; 
    // we use the default alpha=0.7, final_top_k=50

    // ====== Layer 6: 灞傛绱㈠紩 + 绮炬帓 + 鐢熸垚 ======
    GElementPtr parent_lookup, ce1, ce2, generator;
    pipeline->registerGElement<ParentLookupNode>(&parent_lookup, {fusion}, "ParentLookup");
    pipeline->registerGElement<CrossEncoderNode>(&ce1, {parent_lookup}, "CrossEncoder_1");
    pipeline->registerGElement<CrossEncoderNode>(&ce2, {parent_lookup}, "CrossEncoder_2");
    pipeline->registerGElement<LLMGeneratorNode>(&generator, {ce1, ce2}, "Generator");

    // ====== 鎵ц ======
    CGRAPH_ECHO("[RAG] DAG ready: ~27 nodes, 6 layers. Starting...");
    pipeline->process();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    CGRAPH_ECHO("[RAG] ============================================");
    CGRAPH_ECHO("[RAG] Hierarchical RAG completed in %.1f ms", elapsed);
    CGRAPH_ECHO("[RAG] ============================================");

    GPipelineFactory::destroy(pipeline);
    return 0;
}
