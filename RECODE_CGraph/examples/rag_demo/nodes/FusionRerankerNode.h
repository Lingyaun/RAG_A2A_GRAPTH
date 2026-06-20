#ifndef RAG_FUSION_RERANKER_NODE_H
#define RAG_FUSION_RERANKER_NODE_H

#include "../RAGCommon.h"
#include <map>
#include <algorithm>

class FusionRerankerNode : public GNode {
    int final_top_k_   = 50;
    float alpha_       = 0.7f;

public:
    FusionRerankerNode() { this->setName("FusionReranker"); }
    FusionRerankerNode(int k, float a = 0.7f) : final_top_k_(k), alpha_(a) {
        this->setName("FusionReranker");
    }

    CSTATUS run() override {
        auto* sp = this->getGParam<SearchParam>("search");
        auto* bp = this->getGParam<BM25Param>("bm25");
        if (!sp || !bp) return STATUS_ERR;

        std::map<int, float> merged;
        int total_input = 0;

        // 1. Read Dense results
        { CGRAPH_PARAM_READ_REGION(sp) {
            for (auto& track : sp->top_k) {
                total_input += (int)track.size();
                for (auto& [idx, score] : track) {
                    if (!merged.count(idx) || score > merged[idx])
                        merged[idx] = score;
                }
            }
        }}

        float dense_max = 0.0f;
        for (auto& [idx, score] : merged)
            if (score > dense_max) dense_max = score;

        // 2. Read BM25 results
        std::map<int, float> bm25_map;
        float bm25_max = 1.0f;
        size_t bm25_count = 0;
        { CGRAPH_PARAM_READ_REGION(bp) {
            bm25_count = bp->bm25_results.size();
            for (auto& [idx, score] : bp->bm25_results) {
                bm25_map[idx] = score;
                if (score > bm25_max) bm25_max = score;
            }
        }}

        // 3. Weighted fusion
        std::map<int, float> final_scores;
        for (auto& [idx, score] : merged) {
            float norm_score = dense_max > 0 ? score / dense_max : 1.0f;
            final_scores[idx] = alpha_ * norm_score;
        }
        for (auto& [idx, score] : bm25_map) {
            float norm_score = bm25_max > 0 ? score / bm25_max : 1.0f;
            float bm25_contrib = (1.0f - alpha_) * norm_score;
            if (final_scores.count(idx)) final_scores[idx] += bm25_contrib;
            else final_scores[idx] = bm25_contrib;
        }

        std::vector<std::pair<int, float>> ranked;
        for (auto& kv : final_scores) ranked.push_back(kv);
        std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if ((int)ranked.size() > final_top_k_) ranked.resize(final_top_k_);

        auto* fp = this->getGParam<FusionParam>("fusion");
        CGRAPH_PARAM_WRITE_REGION(fp) { fp->fused_results = ranked; }

        int bm25_contrib = 0;
        for (auto& [idx, _] : ranked)
            if (bm25_map.count(idx)) ++bm25_contrib;

        CGRAPH_ECHO("[RAG] FusionReranker: %d items (%d dense + %zu bm25) -> %zu fused (alpha=%.2f, bm25=%d)",
                    total_input, total_input - (int)bm25_count, bm25_count, ranked.size(), alpha_, bm25_contrib);
        return STATUS_OK;
    }
};

#endif