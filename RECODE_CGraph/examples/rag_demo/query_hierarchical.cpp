// ============================================================
// query_hierarchical.cpp — 层次索引 + 混合检索 + 多级漏斗 RAG
//
// 完整 DAG (约 40 节点, 7 层并行窗口):
//
// Layer 1: Loader → Chunker → EmbedCompute → EmbedMerge (串行建库)
// Layer 2: Setup → Decomposer (查询初始化)
// Layer 3: 4×QEmbedCompute → 4×QEmbedMerge (并行)
// Layer 4: 8×SearchCompute→SearchMerge + BM25Compute→BM25Merge (10路并行)
// Layer 5: FusionCompute → FusionMerge
// Layer 6: ParentCompute → ParentMerge
// Layer 7: 2×CECompute → 2×CEMerge (并行精排)
// Layer 8: PromptBuild → AnswerMerge
//
// 所有节点细粒度拆分: Compute 只做 READ+计算, Merge 只做 WRITE
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
    CGRAPH_ECHO("[RAG] HIERARCHICAL RAG (Phase 6 + Fine-Grained)");
    CGRAPH_ECHO("[RAG] Hierarchy: Small chunks (index) -> Parent paragraphs (LLM)");
    CGRAPH_ECHO("[RAG] Hybrid: BM25 + Dense alpha=0.7");
    CGRAPH_ECHO("[RAG] Funnel: Recall 200 -> Filter 50 -> Rerank 5");
    CGRAPH_ECHO("[RAG] Nodes: ~40 fine-grained (Compute+Merge pairs)");
    CGRAPH_ECHO("[RAG] %zu file(s), Query: %s",
                file_paths.size(), question.c_str());
    CGRAPH_ECHO("[RAG] ============================================");

    auto t_start = std::chrono::high_resolution_clock::now();

    GPipelinePtr pipeline = GPipelineFactory::create();

    // ====== Layer 1: 建库 (4节点, 串行) ======
    GElementPtr init, loader, chunker;
    GElementPtr emb_comp, emb_merge;
    auto emb_buf = std::make_shared<EmbedIntermediate>();

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<DocLoaderNode>(&loader, {init}, "DocLoader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedComputeNode>(&emb_comp, {chunker}, "EmbedCompute");
    pipeline->registerGElement<EmbedMergeNode>(&emb_merge, {emb_comp}, "EmbedMerge");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);
    dynamic_cast<EmbedComputeNode*>(emb_comp)->setBuffer(emb_buf);
    dynamic_cast<EmbedMergeNode*>(emb_merge)->setBuffer(emb_buf);

    // ====== Layer 2: 查询初始化 (2节点, 串行) ======
    GElementPtr setup, decomposer;
    pipeline->registerGElement<QuerySetupNode>(&setup, {emb_merge}, "QuerySetup");
    pipeline->registerGElement<QueryDecomposerNode>(&decomposer, {setup}, "Decomposer");
    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);

    // ====== Layer 3: 4路并行 Query Embedding (Compute+Merge) ======
    GElementPtr qec[4], qem[4];
    std::shared_ptr<QEmbedIntermediate> qbuf[4];
    for (int i = 0; i < 4; ++i) {
        qbuf[i] = std::make_shared<QEmbedIntermediate>();
        char nm[16];
        snprintf(nm, sizeof(nm), "QEC%d", i);
        pipeline->registerGElement<QueryEmbedComputeNode>(&qec[i], {decomposer}, nm);
        dynamic_cast<QueryEmbedComputeNode*>(qec[i])->setSubQueryIndex(i);
        dynamic_cast<QueryEmbedComputeNode*>(qec[i])->setBuffer(qbuf[i]);

        snprintf(nm, sizeof(nm), "QEM%d", i);
        pipeline->registerGElement<QueryEmbedMergeNode>(&qem[i], {qec[i]}, nm);
        dynamic_cast<QueryEmbedMergeNode*>(qem[i])->setBuffer(qbuf[i]);
    }

    // ====== Layer 4: 8路Dense + 1路BM25 (Compute+Merge, 10路并行) ======
    GElementPtr sc[4][2], sm[4][2];
    std::shared_ptr<SearchIntermediate> sbuf[4][2];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 2; ++j) {
        sbuf[i][j] = std::make_shared<SearchIntermediate>();
        char nm[16];
        snprintf(nm, sizeof(nm), "SC%d%c", i, 'A'+j);
        pipeline->registerGElement<SearchComputeNode>(&sc[i][j], {qem[i]}, nm);
        dynamic_cast<SearchComputeNode*>(sc[i][j])->configure(25, j, 2, i);
        dynamic_cast<SearchComputeNode*>(sc[i][j])->setBuffer(sbuf[i][j]);

        snprintf(nm, sizeof(nm), "SM%d%c", i, 'A'+j);
        pipeline->registerGElement<SearchMergeNode>(&sm[i][j], {sc[i][j]}, nm);
        dynamic_cast<SearchMergeNode*>(sm[i][j])->setBuffer(sbuf[i][j]);
    }

    // 1路 BM25 (Compute+Merge)
    GElementPtr bmc, bmm;
    auto bmbuf = std::make_shared<BM25Intermediate>();
    pipeline->registerGElement<BM25ComputeNode>(&bmc, {emb_merge}, "BM25Compute");
    pipeline->registerGElement<BM25MergeNode>(&bmm, {bmc}, "BM25Merge");
    dynamic_cast<BM25ComputeNode*>(bmc)->configure(50);
    dynamic_cast<BM25ComputeNode*>(bmc)->setBuffer(bmbuf);
    dynamic_cast<BM25MergeNode*>(bmm)->setBuffer(bmbuf);

    // ====== Layer 5: 混合融合 (Compute+Merge) ======
    GElementPtr fc, fm;
    auto fbuf = std::make_shared<FusionIntermediate>();
    std::set<GElement*> fusion_deps = {
        sm[0][0], sm[0][1], sm[1][0], sm[1][1],
        sm[2][0], sm[2][1], sm[3][0], sm[3][1], bmm
    };
    pipeline->registerGElement<FusionComputeNode>(&fc, fusion_deps, "FusionCompute");
    pipeline->registerGElement<FusionMergeNode>(&fm, {fc}, "FusionMerge");
    dynamic_cast<FusionComputeNode*>(fc)->setBuffer(fbuf);
    dynamic_cast<FusionMergeNode*>(fm)->setBuffer(fbuf);

    // ====== Layer 6: 层次索引 ParentLookup (Compute+Merge) ======
    GElementPtr pc, pm;
    auto pbuf = std::make_shared<ParentIntermediate>();
    pipeline->registerGElement<ParentComputeNode>(&pc, {fm}, "ParentCompute");
    pipeline->registerGElement<ParentMergeNode>(&pm, {pc}, "ParentMerge");
    dynamic_cast<ParentComputeNode*>(pc)->setBuffer(pbuf);
    dynamic_cast<ParentMergeNode*>(pm)->setBuffer(pbuf);

    // ====== Layer 7: 2路并行 CrossEncoder 精排 (Compute+Merge) ======
    GElementPtr cec[2], cem[2];
    std::shared_ptr<CEIntermediate> cebuf[2];
    for (int i = 0; i < 2; ++i) {
        cebuf[i] = std::make_shared<CEIntermediate>();
        char nm[16];
        snprintf(nm, sizeof(nm), "CEC%d", i+1);
        pipeline->registerGElement<CEComputeNode>(&cec[i], {pm}, nm);
        dynamic_cast<CEComputeNode*>(cec[i])->setBuffer(cebuf[i]);

        snprintf(nm, sizeof(nm), "CEM%d", i+1);
        pipeline->registerGElement<CEMergeNode>(&cem[i], {cec[i]}, nm);
        dynamic_cast<CEMergeNode*>(cem[i])->setBuffer(cebuf[i]);
    }

    // ====== Layer 8: 生成 (PromptBuild+AnswerMerge) ======
    GElementPtr pb, am;
    auto ans_buf = std::make_shared<AnswerIntermediate>();
    pipeline->registerGElement<PromptBuildNode>(&pb, {cem[0], cem[1]}, "PromptBuild");
    pipeline->registerGElement<AnswerMergeNode>(&am, {pb}, "AnswerMerge");
    dynamic_cast<PromptBuildNode*>(pb)->setBuffer(ans_buf);
    dynamic_cast<AnswerMergeNode*>(am)->setBuffer(ans_buf);

    // ====== 执行 ======
    CGRAPH_ECHO("[RAG] DAG ready: ~40 nodes, 8 layers. Starting...");
    pipeline->process();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    CGRAPH_ECHO("[RAG] ============================================");
    CGRAPH_ECHO("[RAG] Hierarchical RAG completed in %.1f ms", elapsed);
    CGRAPH_ECHO("[RAG] ============================================");

    GPipelineFactory::destroy(pipeline);
    return 0;
}
