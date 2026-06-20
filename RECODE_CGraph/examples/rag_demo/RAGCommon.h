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
// 细粒度 Param 拆分 — 每个阶段独立 shared_mutex，最大化并行度
//
// 阶段对照:
//   DocParam       — 文档加载+切块
//   EmbedParam     — 文档向量化
//   QueryParam     — 查询设置+分解
//   QueryEmbedParam— 查询向量化
//   SearchParam    — Dense检索结果
//   BM25Param      — BM25检索结果
//   FusionParam    — 融合结果
//   ParentParam    — 层次映射结果
//   AnswerParam    — 最终回答
// ============================================================

// ===== 工具函数 =====

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

inline void trim_inplace(std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a]==' '||s[a]=='\t'||s[a]=='\n'||s[a]=='\r')) ++a;
    while (b > a && (s[b-1]==' '||s[b-1]=='\t'||s[b-1]=='\n'||s[b-1]=='\r')) --b;
    s = s.substr(a, b - a);
}

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

// ===== 阶段1: 文档加载与切块 =====
struct DocParam : public GParam {
    void reset() override {
        documents.clear();
        chunks_small.clear();
        chunks_large.clear();
        small_to_parent.clear();
        chunks.clear();
    }
    std::vector<std::string> documents;
    std::vector<std::string> chunks_small;
    std::vector<std::string> chunks_large;
    std::vector<int> small_to_parent;
    std::vector<std::string> chunks;            // 兼容 = chunks_small
};

// ===== 阶段2: 文档向量化 =====
struct EmbedParam : public GParam {
    void reset() override {
        embeddings.clear();
        embeddings_small.clear();
    }
    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<float>> embeddings_small;
};

// ===== 阶段3: 查询设置 =====
struct QueryParam : public GParam {
    void reset() override {
        query.clear();
        sub_queries.clear();
    }
    std::string query;
    std::vector<std::string> sub_queries;
};

// ===== 阶段4: 查询向量化 =====
struct QueryEmbedParam : public GParam {
    void reset() override { query_embeddings.clear(); }
    std::vector<std::vector<float>> query_embeddings;
};

// ===== 阶段5: Dense检索 =====
struct SearchParam : public GParam {
    void reset() override { top_k.clear(); }
    std::vector<std::vector<std::pair<int, float>>> top_k;
};

// ===== 阶段6: BM25检索 =====
struct BM25Param : public GParam {
    void reset() override { bm25_results.clear(); }
    std::vector<std::pair<int, float>> bm25_results;
};

// ===== 阶段7: 融合 =====
struct FusionParam : public GParam {
    void reset() override { fused_results.clear(); }
    std::vector<std::pair<int, float>> fused_results;
};

// ===== 阶段8: 层次映射 =====
struct ParentParam : public GParam {
    void reset() override { parent_chunks.clear(); }
    std::vector<std::pair<int, float>> parent_chunks;
};

// ===== 阶段9: 最终回答 =====
struct AnswerParam : public GParam {
    void reset() override { answer.clear(); }
    std::string answer;
};

// ===== InitNode: 统一创建所有 Param =====
class InitNode : public GNode {
public:
    InitNode() { this->setName("Init"); }
    CSTATUS init() override {
        auto st = this->createGParam<DocParam>("doc");
        if (st != STATUS_OK) return st;
        st = this->createGParam<EmbedParam>("embed");
        if (st != STATUS_OK) return st;
        st = this->createGParam<QueryParam>("query");
        if (st != STATUS_OK) return st;
        st = this->createGParam<QueryEmbedParam>("qembed");
        if (st != STATUS_OK) return st;
        st = this->createGParam<SearchParam>("search");
        if (st != STATUS_OK) return st;
        st = this->createGParam<BM25Param>("bm25");
        if (st != STATUS_OK) return st;
        st = this->createGParam<FusionParam>("fusion");
        if (st != STATUS_OK) return st;
        st = this->createGParam<ParentParam>("parent");
        if (st != STATUS_OK) return st;
        return this->createGParam<AnswerParam>("answer");
    }
    CSTATUS run() override { return STATUS_OK; }
};

#endif