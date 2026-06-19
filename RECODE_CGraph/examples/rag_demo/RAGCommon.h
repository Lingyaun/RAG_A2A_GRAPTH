#ifndef RAG_COMMON_H
#define RAG_COMMON_H

#include "../../src/GraphCtrl/GraphInclude.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

// ============================================================
// RAGParam — 整个 Pipeline 中共享的参数
// 和test/MyGParam/MyParam1.h 写法完全相同，字段换成RAG语义
// ============================================================
struct RAGParam : public GParam {
    void reset() override {
        documents.clear();
        chunks.clear();
        embeddings.clear();
        query.clear();
        sub_queries.clear();
        query_embeddings.clear();
        top_k.clear();
        fused_results.clear();
        answer.clear();
    }

    // 建库阶段
    std::vector<std::string> documents;
    std::vector<std::string> chunks;
    std::vector<std::vector<float>> embeddings;

    // 查询阶段
    std::string query;
    std::vector<std::string> sub_queries;
    std::vector<std::vector<float>> query_embeddings;

    // 检索结果：每路检索一组 <chunk_index, score>
    std::vector<std::vector<std::pair<int, float>>> top_k;

    // 融合后的最终 Top-K
    std::vector<std::pair<int, float>> fused_results;

    // 最终回答
    std::string answer;
};

// ============================================================
// 通用工具
// ============================================================

// 余弦相似度（纯 std::vector<float>，不依赖任何第三方库）
inline float cosine_similarity(const std::vector<float>& a,
                               const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f) return 0.0f;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// 按分隔符拆分字符串
inline std::vector<std::string> split_string(const std::string& text,
                                              char delimiter) {
    std::vector<std::string> result;
    size_t start = 0, end;
    while ((end = text.find(delimiter, start)) != std::string::npos) {
        auto token = text.substr(start, end - start);
        if (!token.empty()) result.push_back(token);
        start = end + 1;
    }
    auto last = text.substr(start);
    if (!last.empty()) result.push_back(last);
    return result;
}

#endif
