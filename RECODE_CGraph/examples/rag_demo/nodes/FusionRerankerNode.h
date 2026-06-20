#ifndef RAG_FUSION_RERANKER_NODE_H
#define RAG_FUSION_RERANKER_NODE_H

#include "../RAGCommon.h"
#include <map>
#include <algorithm>

// ===== Intermediate =====
struct FusionIntermediate {
    std::vector<std::pair<int, float>> fused;
};

// ===== FusionComputeNode: READ SearchParam + BM25Param, COMPUTE 加权融合 =====
class FusionComputeNode : public GNode {
    int final_top_k_   = 50;
    float alpha_       = 0.7f;

public:
    FusionComputeNode() { this->setName("FusionCompute"); }
    FusionComputeNode(int k, float a = 0.7f) : final_top_k_(k), alpha_(a) {
        this->setName("FusionCompute");
    }
    void setBuffer(std::shared_ptr<FusionIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* sp = this->getGParam<SearchParam>("search");
        auto* bp = this->getGParam<BM25Param>("bm25");
        if (!sp || !bp) return STATUS_ERR;

        std::map<int, float> merged;
        int total_input = 0;

        { CGRAPH_PARAM_READ_REGION(sp) {
            for (auto& track : sp->top_k) {
                total_input += (int)track.size();
                for (auto& [idx, score] : track)
                    if (!merged.count(idx) || score > merged[idx])
                        merged[idx] = score;
            }
        }}

        float dense_max = 0.0f;
        for (auto& [_k, score] : merged)
            if (score > dense_max) dense_max = score;

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

        int bm25_contrib = 0;
        for (auto& [idx, _] : ranked)
            if (bm25_map.count(idx)) ++bm25_contrib;

        buf_->fused = std::move(ranked);
        CGRAPH_ECHO("[RAG] FusionCompute: %d dense + %zu bm25 -> %zu (alpha=%.2f, bm25=%d)",
            total_input, bm25_count, buf_->fused.size(), alpha_, bm25_contrib);
        return STATUS_OK;
    }

private:
    std::shared_ptr<FusionIntermediate> buf_;
};

// ===== FusionMergeNode: READ 中间结果, WRITE FusionParam =====
class FusionMergeNode : public GNode {
public:
    FusionMergeNode() { this->setName("FusionMerge"); }
    void setBuffer(std::shared_ptr<FusionIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* fp = this->getGParam<FusionParam>("fusion");
        if (!fp) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(fp) { fp->fused_results = std::move(buf_->fused); }
        CGRAPH_ECHO("[RAG] FusionMerge: committed %zu items", fp->fused_results.size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<FusionIntermediate> buf_;
};

#endif
