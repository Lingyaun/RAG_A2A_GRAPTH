#ifndef RAG_FUSION_RERANKER_NODE_H
#define RAG_FUSION_RERANKER_NODE_H

#include "../RAGCommon.h"
#include <map>
#include <algorithm>

// ============================================================
// FusionRerankerNode — 多路检索结果融合 + 混合检索
//
// Phase 6 扩展：同时对接 Dense 检索结果 (top_k) 和 BM25 结果 (bm25_results)
// 加权融合: final_score = α * dense_norm + (1-α) * bm25_norm
// 默认 α=0.7（偏向语义，保留关键词能力）
//
// 输入: top_k (Dense results) + bm25_results (BM25 results)
// 输出: fused_results (Top-50 small chunk IDs)
// ============================================================
class FusionRerankerNode : public GNode {
    int final_top_k_   = 50;   // 输出候选数
    float alpha_       = 0.7f; // Dense 权重

public:
    FusionRerankerNode() { this->setName("FusionReranker"); }
    FusionRerankerNode(int k, float a = 0.7f) : final_top_k_(k), alpha_(a) {
        this->setName("FusionReranker");
    }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        std::map<int, float> merged;
        int total_input = 0;

        // ====== 1. 读取 Dense 结果（来自 VectorSearchNode） ======
        {
            CGRAPH_PARAM_READ_REGION(p) {
                for (auto& track : p->top_k) {
                    total_input += (int)track.size();
                    for (auto& [idx, score] : track) {
                        if (!merged.count(idx) || score > merged[idx])
                            merged[idx] = score;
                    }
                }
            }
        }

        // ====== 2. 归一化 dense scores ======
        float dense_max = 0.0f;
        for (auto& [idx, score] : merged) {
            if (score > dense_max) dense_max = score;
        }

        // ====== 3. 读取 BM25 结果并归一化 ======
        std::map<int, float> bm25_map;
        float bm25_max = 1.0f;
        {
            CGRAPH_PARAM_READ_REGION(p) {
                for (auto& [idx, score] : p->bm25_results) {
                    bm25_map[idx] = score;
                    if (score > bm25_max) bm25_max = score;
                }
            }
        }

        // ====== 4. 加权融合 ======
        std::map<int, float> final_scores;

        // 4a. 融合 Dense 分数
        for (auto& [idx, score] : merged) {
            float norm_score = dense_max > 0 ? score / dense_max : 1.0f;
            final_scores[idx] = alpha_ * norm_score;
        }

        // 4b. 融合 BM25 分数
        for (auto& [idx, score] : bm25_map) {
            float norm_score = bm25_max > 0 ? score / bm25_max : 1.0f;
            float bm25_contrib = (1.0f - alpha_) * norm_score;
            if (final_scores.count(idx)) {
                final_scores[idx] += bm25_contrib;
            } else {
                final_scores[idx] = bm25_contrib;
            }
        }

        // ====== 5. 排序取 Top-K ======
        std::vector<std::pair<int, float>> ranked;
        for (auto& kv : final_scores) ranked.push_back(kv);
        std::sort(ranked.begin(), ranked.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        if ((int)ranked.size() > final_top_k_) ranked.resize(final_top_k_);

        CGRAPH_PARAM_WRITE_REGION(p) { p->fused_results = ranked; }

        int bm25_contrib = 0;
        for (auto& [idx, _] : ranked) {
            if (bm25_map.count(idx)) ++bm25_contrib;
        }

        CGRAPH_ECHO("[RAG] FusionReranker: %d items (%d dense + %zu bm25) → %zu fused (α=%.2f, bm25=%d)",
                    total_input, total_input - (int)p->bm25_results.size(),
                    p->bm25_results.size(), ranked.size(), alpha_, bm25_contrib);
        return STATUS_OK;
    }
};

#endif
