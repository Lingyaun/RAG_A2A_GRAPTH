#ifndef RAG_COMMON_H
#define RAG_COMMON_H

#include "../../src/GraphCtrl/GraphInclude.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cctype>

// ============================================================
// RAGParam — 整个 Pipeline 中共享的参数
// Phase 5 扩展：双粒度chunk（大chunk=父段落, 小chunk=索引用）
// Phase 6 扩展：bm25_results, parent_chunks
// ============================================================
struct RAGParam : public GParam {
    void reset() override {
        documents.clear();
        chunks.clear();           // 保留兼容：指向 chunks_small
        chunks_small.clear();
        chunks_large.clear();
        small_to_parent.clear();
        embeddings.clear();
        embeddings_small.clear();
        query.clear();
        sub_queries.clear();
        query_embeddings.clear();
        top_k.clear();
        fused_results.clear();
        bm25_results.clear();
        parent_chunks.clear();
        answer.clear();
    }

    // 建库阶段
    std::vector<std::string> documents;
    std::vector<std::string> chunks;              // Phase 5 保留兼容 = chunks_small
    std::vector<std::string> chunks_small;        // 小chunk（索引用，128-256字符）
    std::vector<std::string> chunks_large;        // 大chunk（父段落，LLM用，512-1024字符）
    std::vector<int> small_to_parent;             // small[i] → large[small_to_parent[i]]
    std::vector<std::vector<float>> embeddings;   // 保留兼容 = embeddings_small
    std::vector<std::vector<float>> embeddings_small; // 小chunk的embedding

    // 查询阶段
    std::string query;
    std::vector<std::string> sub_queries;
    std::vector<std::vector<float>> query_embeddings;

    // 检索结果：每路检索一组 <chunk_index, score>
    std::vector<std::vector<std::pair<int, float>>> top_k;

    // BM25 检索结果（Phase 6）
    std::vector<std::pair<int, float>> bm25_results;

    // 融合后的最终 Top-K（small chunk IDs）
    std::vector<std::pair<int, float>> fused_results;

    // 层次索引（Phase 6）：小chunk → 父段落映射后的结果
    std::vector<std::pair<int, float>> parent_chunks;

    // 最终回答
    std::string answer;
};

// ============================================================
// 通用工具
// ============================================================

// 余弦相似度
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

// 按字符分隔符拆分
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

// 按字符串分隔符拆分
// 原地去首尾空白
inline void trim_inplace(std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a]==' '||s[a]=='\t'||s[a]=='\n'||s[a]=='\r')) ++a;
    while (b > a && (s[b-1]==' '||s[b-1]=='\t'||s[b-1]=='\n'||s[b-1]=='\r')) --b;
    s = s.substr(a, b - a);
}

// 按字符串分隔符拆分
inline std::vector<std::string> split_by_string(const std::string& text,
                                                 const std::string& sep) {
    std::vector<std::string> result;
    if (sep.empty()) return {text};
    size_t start = 0;
    while (start < text.size()) {
        size_t pos = text.find(sep, start);
        if (pos == std::string::npos) {
            auto part = text.substr(start);
            trim_inplace(part);
            if (!part.empty()) result.push_back(part);
            break;
        }
        auto part = text.substr(start, pos - start);
        trim_inplace(part);
        if (!part.empty()) result.push_back(part);
        start = pos + sep.size();
    }
    return result;
}


// 简单英文分词 + 去停用词（用于 BM25）
inline std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : text) {
        if (std::isalnum((unsigned char)c)) {
            cur += (char)std::tolower((unsigned char)c);
        } else {
            if (cur.size() >= 2) tokens.push_back(cur);
            cur.clear();
        }
    }
    if (cur.size() >= 2) tokens.push_back(cur);
    return tokens;
}

#endif
