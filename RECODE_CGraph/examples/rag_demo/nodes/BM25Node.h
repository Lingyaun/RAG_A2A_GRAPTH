#ifndef RAG_BM25_NODE_H
#define RAG_BM25_NODE_H

#include "../RAGCommon.h"
#include <map>
#include <cmath>

// ============================================================
// BM25Node — 稀疏检索（关键词匹配）
//
// 对 small chunks 做 BM25 检索，与 Dense 检索互补
// 使用简单的 TF-IDF 风格实现（无外部依赖）
//
// 输出: RAGParam::bm25_results
// ============================================================
class BM25Node : public GNode {
    int top_k_      = 50;    // 粗召回数
    float k1_       = 1.2f;  // BM25 参数
    float b_        = 0.75f; // BM25 参数

public:
    BM25Node() { this->setName("BM25"); }
    BM25Node(int k) : top_k_(k) { this->setName("BM25"); }

    void configure(int k) { top_k_ = k; }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        // 读取 small chunks
        std::vector<std::string> docs;
        {
            CGRAPH_PARAM_READ_REGION(p) {
                docs = p->chunks_small.empty() ? p->chunks : p->chunks_small;
            }
        }
        if (docs.empty()) {
            CGRAPH_PARAM_WRITE_REGION(p) { p->bm25_results.clear(); }
            CGRAPH_ECHO("[RAG] BM25: no chunks to search");
            return STATUS_OK;
        }

        // 读取 query
        std::string q;
        {
            CGRAPH_PARAM_READ_REGION(p) { q = p->query; }
        }
        auto q_tokens = tokenize(q);
        if (q_tokens.empty()) {
            CGRAPH_PARAM_WRITE_REGION(p) { p->bm25_results.clear(); }
            return STATUS_OK;
        }

        // 建立倒排索引: word → {doc_id → term_freq}
        std::unordered_map<std::string, std::unordered_map<int, int>> index;
        std::vector<int> doc_lengths(docs.size());
        int total_docs = (int)docs.size();
        double avg_dl = 0.0;

        for (int i = 0; i < total_docs; ++i) {
            auto toks = tokenize(docs[i]);
            doc_lengths[i] = (int)toks.size();
            avg_dl += doc_lengths[i];
            for (auto& t : toks) {
                index[t][i]++;
            }
        }
        avg_dl /= total_docs;

        // 计算每篇文档的 BM25 分数
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
            if (score > 0.0f)
                scores.push_back({i, score});
        }

        std::sort(scores.begin(), scores.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        if ((int)scores.size() > top_k_)
            scores.resize(top_k_);

        CGRAPH_PARAM_WRITE_REGION(p) { p->bm25_results = scores; }

        float best = scores.empty() ? 0.0f : scores[0].second;
        CGRAPH_ECHO("[RAG] BM25: %d docs → %zu hits, top1=%.4f",
                    total_docs, scores.size(), best);
        return STATUS_OK;
    }
};

#endif
