#ifndef RAG_BM25_NODE_H
#define RAG_BM25_NODE_H

#include "../RAGCommon.h"
#include <map>
#include <cmath>

// ===== Intermediate =====
struct BM25Intermediate {
    std::vector<std::pair<int, float>> scores;
};

// ===== BM25ComputeNode: READ DocParam + QueryParam, COMPUTE BM25 =====
class BM25ComputeNode : public GNode {
    int top_k_      = 50;
    float k1_       = 1.2f;
    float b_        = 0.75f;

public:
    BM25ComputeNode() { this->setName("BM25Compute"); }
    BM25ComputeNode(int k) : top_k_(k) { this->setName("BM25Compute"); }
    void configure(int k) { top_k_ = k; }
    void setBuffer(std::shared_ptr<BM25Intermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* dp = this->getGParam<DocParam>("doc");
        if (!dp) return STATUS_ERR;
        std::vector<std::string> docs;
        { CGRAPH_PARAM_READ_REGION(dp) { docs = dp->chunks_small.empty() ? dp->chunks : dp->chunks_small; }}
        if (docs.empty()) { buf_->scores.clear(); return STATUS_OK; }

        auto* qp = this->getGParam<QueryParam>("query");
        if (!qp) return STATUS_ERR;
        std::string q;
        { CGRAPH_PARAM_READ_REGION(qp) { q = qp->query; }}
        auto q_tokens = tokenize(q);
        if (q_tokens.empty()) { buf_->scores.clear(); return STATUS_OK; }

        std::unordered_map<std::string, std::unordered_map<int, int>> index;
        std::vector<int> doc_lengths(docs.size());
        int total_docs = (int)docs.size();
        double avg_dl = 0.0;
        for (int i = 0; i < total_docs; ++i) {
            auto toks = tokenize(docs[i]);
            doc_lengths[i] = (int)toks.size();
            avg_dl += doc_lengths[i];
            for (auto& t : toks) index[t][i]++;
        }
        avg_dl /= total_docs;

        std::vector<std::pair<int, float>> scores;
        for (int i = 0; i < total_docs; ++i) {
            float score = 0.0f;
            for (auto& qt : q_tokens) {
                auto it = index.find(qt);
                if (it == index.end()) continue;
                auto& postings = it->second;
                auto pit = postings.find(i);
                if (pit == postings.end()) continue;
                int tf = pit->second;
                int df = (int)postings.size();
                float idf = (float)std::log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                float numerator = tf * (k1_ + 1.0f);
                float denominator = tf + k1_ * (1.0f - b_ + b_ * doc_lengths[i] / avg_dl);
                score += idf * numerator / denominator;
            }
            if (score > 0.0f) scores.push_back({i, score});
        }
        std::sort(scores.begin(), scores.end(), [](auto& a, auto& b) { return a.second > b.second; });
        if ((int)scores.size() > top_k_) scores.resize(top_k_);
        buf_->scores = std::move(scores);
        return STATUS_OK;
    }

private:
    std::shared_ptr<BM25Intermediate> buf_;
};

// ===== BM25MergeNode: READ 中间结果, WRITE BM25Param =====
class BM25MergeNode : public GNode {
public:
    BM25MergeNode() { this->setName("BM25Merge"); }
    void setBuffer(std::shared_ptr<BM25Intermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* bm = this->getGParam<BM25Param>("bm25");
        if (!bm) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(bm) { bm->bm25_results = std::move(buf_->scores); }
        float best = bm->bm25_results.empty() ? 0.0f : bm->bm25_results[0].second;
        CGRAPH_ECHO("[RAG] BM25Merge: %zu hits, top1=%.4f", bm->bm25_results.size(), best);
        return STATUS_OK;
    }

private:
    std::shared_ptr<BM25Intermediate> buf_;
};

#endif
