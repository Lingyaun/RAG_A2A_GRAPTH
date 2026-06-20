#ifndef RAG_CROSS_ENCODER_NODE_H
#define RAG_CROSS_ENCODER_NODE_H

#include "../RAGCommon.h"

// ===== Intermediate =====
struct CEIntermediate {
    std::vector<std::pair<int, float>> reranked;
};

// ===== CEComputeNode: READ ParentParam + DocParam + QueryParam, COMPUTE rerank =====
class CEComputeNode : public GNode {
    int final_top_k_ = 5;

public:
    CEComputeNode() { this->setName("CECompute"); }
    CEComputeNode(int k) : final_top_k_(k) { this->setName("CECompute"); }
    void setBuffer(std::shared_ptr<CEIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* pp = this->getGParam<ParentParam>("parent");
        auto* dp = this->getGParam<DocParam>("doc");
        auto* qp = this->getGParam<QueryParam>("query");
        if (!pp || !dp || !qp) return STATUS_ERR;

        std::vector<std::pair<int, float>> candidates;
        std::vector<std::string> large_chunks;
        std::string q;

        { CGRAPH_PARAM_READ_REGION(pp) { candidates = pp->parent_chunks; } }
        { CGRAPH_PARAM_READ_REGION(dp) { large_chunks = dp->chunks_large; } }
        { CGRAPH_PARAM_READ_REGION(qp) { q = qp->query; } }

        if (candidates.empty()) {
            CGRAPH_ECHO("[RAG] CECompute: no candidates");
            buf_->reranked.clear();
            return STATUS_OK;
        }

        auto q_tokens = tokenize(q);
        for (auto& [pid, old_score] : candidates) {
            if (pid < 0 || pid >= (int)large_chunks.size()) continue;
            const auto& chunk_text = large_chunks[pid];
            auto c_tokens = tokenize(chunk_text);
            std::unordered_map<std::string, int> qset;
            for (auto& t : q_tokens) qset[t] = 1;
            int overlap = 0;
            for (auto& t : c_tokens) { if (qset.count(t)) ++overlap; }
            int total = (int)(q_tokens.size() + c_tokens.size() - overlap);
            float jaccard = (total > 0) ? (float)overlap / total : 0.0f;
            old_score = 0.7f * old_score + 0.3f * jaccard;
        }

        std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if ((int)candidates.size() > final_top_k_) candidates.resize(final_top_k_);
        buf_->reranked = std::move(candidates);

        float best = buf_->reranked.empty() ? 0.0f : buf_->reranked[0].second;
        CGRAPH_ECHO("[RAG] CECompute: %zu -> top-%d, best=%.4f",
            buf_->reranked.size(), final_top_k_, best);
        return STATUS_OK;
    }

private:
    std::shared_ptr<CEIntermediate> buf_;
};

// ===== CEMergeNode: READ 中间结果, WRITE ParentParam (覆盖为精排结果) =====
class CEMergeNode : public GNode {
public:
    CEMergeNode() { this->setName("CEMerge"); }
    void setBuffer(std::shared_ptr<CEIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* pp = this->getGParam<ParentParam>("parent");
        if (!pp) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(pp) { pp->parent_chunks = std::move(buf_->reranked); }
        CGRAPH_ECHO("[RAG] CEMerge: committed %zu reranked", pp->parent_chunks.size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<CEIntermediate> buf_;
};

#endif
