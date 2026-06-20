// ============================================================
// query_parallel.cpp — 并行 RAG 查询（Phase 2 核心）
//
// DAG: Loader→Chunker→DocEmbed→Setup→Decomposer
//      → 4×QE(并行) → 8×Search(并行) → 2×Rerank(并行) → Generator
// 共 17 个节点，5 层并行窗口
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
        file_paths.push_back("test_docs/cpp_concurrency.txt");
    if (question.empty())
        question = "What are the key differences between C++11 and C++17 concurrency?";

    CGRAPH_ECHO("[RAG] ============================================");
    CGRAPH_ECHO("[RAG] PARALLEL RAG DAG: %zu file(s)", file_paths.size());
    CGRAPH_ECHO("[RAG] Query: %s", question.c_str());
    CGRAPH_ECHO("[RAG] ============================================");

    // EmbedderNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    GPipelinePtr pipeline = GPipelineFactory::create();

    // ====== Layer 1: 建库 + 设置查询 + 分解 ======
    GElementPtr loader, chunker, doc_embed, setup, decomposer;
    pipeline->registerGElement<DocLoaderNode>(&loader, {}, "DocLoader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedderNode>(&doc_embed, {chunker}, "DocEmbedder");
    pipeline->registerGElement<QuerySetupNode>(&setup, {doc_embed}, "QuerySetup");
    pipeline->registerGElement<QueryDecomposerNode>(&decomposer, {setup}, "Decomposer");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);
    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);

    // ====== Layer 2: 4路并行 Embedding ======
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

    // ====== Layer 3: 8路并行检索 ======
    GElementPtr s0a, s0b, s1a, s1b, s2a, s2b, s3a, s3b;
    pipeline->registerGElement<VectorSearchNode>(&s0a, {qe0}, "S0A");
    pipeline->registerGElement<VectorSearchNode>(&s0b, {qe0}, "S0B");
    pipeline->registerGElement<VectorSearchNode>(&s1a, {qe1}, "S1A");
    pipeline->registerGElement<VectorSearchNode>(&s1b, {qe1}, "S1B");
    pipeline->registerGElement<VectorSearchNode>(&s2a, {qe2}, "S2A");
    pipeline->registerGElement<VectorSearchNode>(&s2b, {qe2}, "S2B");
    pipeline->registerGElement<VectorSearchNode>(&s3a, {qe3}, "S3A");
    pipeline->registerGElement<VectorSearchNode>(&s3b, {qe3}, "S3B");

    dynamic_cast<VectorSearchNode*>(s0a)->configure(10, 0, 2, 0);
    dynamic_cast<VectorSearchNode*>(s0b)->configure(10, 1, 2, 0);
    dynamic_cast<VectorSearchNode*>(s1a)->configure(10, 0, 2, 1);
    dynamic_cast<VectorSearchNode*>(s1b)->configure(10, 1, 2, 1);
    dynamic_cast<VectorSearchNode*>(s2a)->configure(10, 0, 2, 2);
    dynamic_cast<VectorSearchNode*>(s2b)->configure(10, 1, 2, 2);
    dynamic_cast<VectorSearchNode*>(s3a)->configure(10, 0, 2, 3);
    dynamic_cast<VectorSearchNode*>(s3b)->configure(10, 1, 2, 3);

    // ====== Layer 4: 2路并行融合 ======
    GElementPtr rerank_01, rerank_23;
    pipeline->registerGElement<FusionRerankerNode>(&rerank_01, {s0a, s0b, s1a, s1b}, "Rerank_01");
    pipeline->registerGElement<FusionRerankerNode>(&rerank_23, {s2a, s2b, s3a, s3b}, "Rerank_23");

    // ====== Layer 5: 生成 ======
    GElementPtr generator;
    pipeline->registerGElement<LLMGeneratorNode>(&generator, {rerank_01, rerank_23}, "Generator");

    // ====== 执行 ======
    CGRAPH_ECHO("[RAG] DAG ready: 17 nodes, 5 layers. Starting...");
    pipeline->process();
    CGRAPH_ECHO("[RAG] Parallel RAG completed.");

    GPipelineFactory::destroy(pipeline);
    return 0;
}
