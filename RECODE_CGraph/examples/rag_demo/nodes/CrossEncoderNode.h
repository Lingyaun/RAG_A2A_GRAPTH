#ifndef RAG_CROSS_ENCODER_NODE_H
#define RAG_CROSS_ENCODER_NODE_H

#include "../RAGCommon.h"

// ============================================================
// CrossEncoderNode — 多级漏斗精排
//
// 对候选 parent chunks 做 rerank。
// 当前实现：模拟 CrossEncoder 行为（query 与 chunk 的 token 重叠度评分）。
// 生产环境替换为：DashScope/OpenAI Rerank API 或本地 BGE-Reranker。
//
// 输入: RAGParam::parent_chunks（5-15 个 parent IDs）
// 输出: RAGParam::parent_chunks（精排后 Top-5）
// ============================================================
class CrossEncoderNode : public GNode {
    int final_top_k_ = 5;

public:
    CrossEncoderNode() { this->setName("CrossEncoder"); }
    CrossEncoderNode(int k) : final_top_k_(k) { this->setName("CrossEncoder"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        // 读取数据
        std::vector<std::pair<int, float>> candidates;
        std::vector<std::string> large_chunks;
        std::string q;
        {
            CGRAPH_PARAM_READ_REGION(p) {
                candidates = p->parent_chunks;
                large_chunks = p->chunks_large;
                q = p->query;
            }
        }

        if (candidates.empty()) {
            CGRAPH_ECHO("[RAG] CrossEncoder: no candidates");
            return STATUS_OK;
        }

        // Tokenize query once
        auto q_tokens = tokenize(q);

        // 为每个候选重新评分：70% 语义分 + 30% token 重叠分
        for (auto& [pid, old_score] : candidates) {
            if (pid < 0 || pid >= (int)large_chunks.size()) continue;

            const auto& chunk_text = large_chunks[pid];
            auto c_tokens = tokenize(chunk_text);

            // Jaccard 相似度
            std::unordered_map<std::string, int> qset;
            for (auto& t : q_tokens) qset[t] = 1;
            int overlap = 0;
            for (auto& t : c_tokens) {
                if (qset.count(t)) ++overlap;
            }
            int total = (int)(q_tokens.size() + c_tokens.size() - overlap);
            float jaccard = (total > 0) ? (float)overlap / total : 0.0f;

            old_score = 0.7f * old_score + 0.3f * jaccard;
        }

        // 重排并截断
        std::sort(candidates.begin(), candidates.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        if ((int)candidates.size() > final_top_k_)
            candidates.resize(final_top_k_);

        CGRAPH_PARAM_WRITE_REGION(p) { p->parent_chunks = candidates; }

        float best = candidates.empty() ? 0.0f : candidates[0].second;
        CGRAPH_ECHO("[RAG] CrossEncoder: %zu candidates → top-%d, best=%.4f",
                    p->parent_chunks.size(), final_top_k_, best);
        return STATUS_OK;
    }
};

#endif
