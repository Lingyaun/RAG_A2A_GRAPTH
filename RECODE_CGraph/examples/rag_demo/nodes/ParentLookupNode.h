#ifndef RAG_PARENT_LOOKUP_NODE_H
#define RAG_PARENT_LOOKUP_NODE_H

#include "../RAGCommon.h"

// ===== Intermediate =====
struct ParentIntermediate {
    std::vector<std::pair<int, float>> parent_results;
};

// ===== ParentComputeNode: READ FusionParam + DocParam, COMPUTE small_to_parent 映射 =====
class ParentComputeNode : public GNode {
public:
    ParentComputeNode() { this->setName("ParentCompute"); }
    void setBuffer(std::shared_ptr<ParentIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* fp = this->getGParam<FusionParam>("fusion");
        auto* dp = this->getGParam<DocParam>("doc");
        if (!fp || !dp) return STATUS_ERR;

        std::vector<std::pair<int, float>> fused;
        { CGRAPH_PARAM_READ_REGION(fp) { fused = fp->fused_results; }}

        std::unordered_map<int, float> parent_scores;
        { CGRAPH_PARAM_READ_REGION(dp) {
            if (dp->small_to_parent.empty()) {
                buf_->parent_results = fused;
                CGRAPH_ECHO("[RAG] ParentCompute: no map, passthrough %zu", fused.size());
                return STATUS_OK;
            }
            for (auto& [small_id, score] : fused) {
                if (small_id < 0 || small_id >= (int)dp->small_to_parent.size()) continue;
                int parent_id = dp->small_to_parent[small_id];
                auto it = parent_scores.find(parent_id);
                if (it == parent_scores.end() || score > it->second)
                    parent_scores[parent_id] = score;
            }
        }}

        std::vector<std::pair<int, float>> results;
        for (auto& [pid, score] : parent_scores) results.push_back({pid, score});
        std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.second > b.second; });
        buf_->parent_results = std::move(results);
        CGRAPH_ECHO("[RAG] ParentCompute: %zu small -> %zu parents",
            fused.size(), buf_->parent_results.size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<ParentIntermediate> buf_;
};

// ===== ParentMergeNode: READ 中间结果, WRITE ParentParam =====
class ParentMergeNode : public GNode {
public:
    ParentMergeNode() { this->setName("ParentMerge"); }
    void setBuffer(std::shared_ptr<ParentIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* pp = this->getGParam<ParentParam>("parent");
        if (!pp) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(pp) { pp->parent_chunks = std::move(buf_->parent_results); }
        CGRAPH_ECHO("[RAG] ParentMerge: committed %zu parents", pp->parent_chunks.size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<ParentIntermediate> buf_;
};

#endif
