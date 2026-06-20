// ============================================================
// query_parallel.cpp — 并行 RAG 查询（Phase 2 核心）
//
// DAG: Loader→Chunker→EmbedCompute→EmbedMerge→Setup→Decomposer
//      → 4×QEmbedCompute→4×QEmbedMerge(并行)
//      → 8×SearchCompute→8×SearchMerge(并行)
//      → 2×FusionCompute→2×FusionMerge(并行)
//      → PromptBuild→AnswerMerge
// 约 25+ 节点，5 层并行窗口
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

    // EmbedComputeNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    GPipelinePtr pipeline = GPipelineFactory::create();

    // ====== Layer 1: 建库 + 设置查询 + 分解 ======
    GElementPtr init, loader, chunker;
    GElementPtr emb_comp, emb_merge, setup, decomposer;
    auto emb_buf = std::make_shared<EmbedIntermediate>();

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<DocLoaderNode>(&loader, {init}, "DocLoader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedComputeNode>(&emb_comp, {chunker}, "EmbedCompute");
    pipeline->registerGElement<EmbedMergeNode>(&emb_merge, {emb_comp}, "EmbedMerge");
    pipeline->registerGElement<QuerySetupNode>(&setup, {emb_merge}, "QuerySetup");
    pipeline->registerGElement<QueryDecomposerNode>(&decomposer, {setup}, "Decomposer");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);
    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);
    dynamic_cast<EmbedComputeNode*>(emb_comp)->setBuffer(emb_buf);
    dynamic_cast<EmbedMergeNode*>(emb_merge)->setBuffer(emb_buf);

    // ====== Layer 2: 4路并行 Query Embedding (Compute+Merge) ======
    GElementPtr qc[4], qm[4];
    std::shared_ptr<QEmbedIntermediate> qbuf[4];
    for (int i = 0; i < 4; ++i) {
        qbuf[i] = std::make_shared<QEmbedIntermediate>();
        char nm[16];
        snprintf(nm, sizeof(nm), "QEC%d", i);
        pipeline->registerGElement<QueryEmbedComputeNode>(&qc[i], {decomposer}, nm);
        dynamic_cast<QueryEmbedComputeNode*>(qc[i])->setSubQueryIndex(i);
        dynamic_cast<QueryEmbedComputeNode*>(qc[i])->setBuffer(qbuf[i]);

        snprintf(nm, sizeof(nm), "QEM%d", i);
        pipeline->registerGElement<QueryEmbedMergeNode>(&qm[i], {qc[i]}, nm);
        dynamic_cast<QueryEmbedMergeNode*>(qm[i])->setBuffer(qbuf[i]);
    }

    // ====== Layer 3: 8路并行 Search (Compute+Merge) ======
    GElementPtr sc[4][2], sm[4][2];
    std::shared_ptr<SearchIntermediate> sbuf[4][2];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) {
        sbuf[i][j] = std::make_shared<SearchIntermediate>();
        char nm[16];
        snprintf(nm, sizeof(nm), "SC%d%c", i, 'A'+j);
        pipeline->registerGElement<SearchComputeNode>(&sc[i][j], {qm[i]}, nm);
        dynamic_cast<SearchComputeNode*>(sc[i][j])->configure(10, j, 2, i);
        dynamic_cast<SearchComputeNode*>(sc[i][j])->setBuffer(sbuf[i][j]);

        snprintf(nm, sizeof(nm), "SM%d%c", i, 'A'+j);
        pipeline->registerGElement<SearchMergeNode>(&sm[i][j], {sc[i][j]}, nm);
        dynamic_cast<SearchMergeNode*>(sm[i][j])->setBuffer(sbuf[i][j]);
    }

    // ====== Layer 4: 2路并行融合 (Compute+Merge) ======
    GElementPtr fc01, fc23, fm01, fm23;
    auto fbuf01 = std::make_shared<FusionIntermediate>();
    auto fbuf23 = std::make_shared<FusionIntermediate>();

    pipeline->registerGElement<FusionComputeNode>(&fc01,
        {sm[0][0], sm[0][1], sm[1][0], sm[1][1]}, "FC01");
    pipeline->registerGElement<FusionComputeNode>(&fc23,
        {sm[2][0], sm[2][1], sm[3][0], sm[3][1]}, "FC23");
    pipeline->registerGElement<FusionMergeNode>(&fm01, {fc01}, "FM01");
    pipeline->registerGElement<FusionMergeNode>(&fm23, {fc23}, "FM23");

    dynamic_cast<FusionComputeNode*>(fc01)->setBuffer(fbuf01);
    dynamic_cast<FusionComputeNode*>(fc23)->setBuffer(fbuf23);
    dynamic_cast<FusionMergeNode*>(fm01)->setBuffer(fbuf01);
    dynamic_cast<FusionMergeNode*>(fm23)->setBuffer(fbuf23);

    // ====== Layer 5: 生成 (PromptBuild+AnswerMerge) ======
    GElementPtr prompt_build, answer_merge;
    auto ans_buf = std::make_shared<AnswerIntermediate>();

    pipeline->registerGElement<PromptBuildNode>(&prompt_build, {fm01, fm23}, "PromptBuild");
    pipeline->registerGElement<AnswerMergeNode>(&answer_merge, {prompt_build}, "AnswerMerge");

    dynamic_cast<PromptBuildNode*>(prompt_build)->setBuffer(ans_buf);
    dynamic_cast<AnswerMergeNode*>(answer_merge)->setBuffer(ans_buf);

    // ====== 执行 ======
    CGRAPH_ECHO("[RAG] DAG ready: ~25 nodes, 5 layers. Starting...");
    pipeline->process();
    CGRAPH_ECHO("[RAG] Parallel RAG completed.");

    GPipelineFactory::destroy(pipeline);
    return 0;
}
