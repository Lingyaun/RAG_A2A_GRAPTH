#ifndef RAG_EMBEDDER_NODE_H
#define RAG_EMBEDDER_NODE_H

#include "../RAGCommon.h"
#include <EmbeddingClient.h>
#include <random>
#include <memory>

// ===== Intermediate buffers =====
struct EmbedIntermediate {
    std::vector<std::vector<float>> embeddings;
};

struct QEmbedIntermediate {
    std::vector<float> embedding;
};

// ===== Embedding infrastructure (shared) =====
inline std::shared_ptr<EmbeddingClient> g_embed_client;

inline std::vector<float> mock_embed(const std::string& text) {
    std::vector<float> vec(128);
    std::mt19937 rng((unsigned)std::hash<std::string>{}(text));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float norm = 0.0f;
    for (auto& v : vec) { v = dist(rng); norm += v * v; }
    norm = std::sqrt(norm);
    if (norm > 0) for (auto& v : vec) v /= norm;
    return vec;
}

// ===== Doc Embedding: Compute + Merge =====
class EmbedComputeNode : public GNode {
public:
    EmbedComputeNode() { this->setName("EmbedCompute"); }
    void setBuffer(std::shared_ptr<EmbedIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* dp = this->getGParam<DocParam>("doc");
        if (!dp) return STATUS_ERR;
        CGRAPH_PARAM_READ_REGION(dp) {
            auto& src = dp->chunks_small.empty() ? dp->chunks : dp->chunks_small;
            buf_->embeddings = try_real_embed_batch(src);
        }
        CGRAPH_ECHO("[RAG] EmbedCompute: %zu vectors", buf_->embeddings.size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<EmbedIntermediate> buf_;

    std::vector<std::vector<float>> try_real_embed_batch(const std::vector<std::string>& texts) {
        if (texts.empty()) return {};
        if (g_embed_client && g_embed_client->is_configured()) {
            try {
                auto v = g_embed_client->embed(texts);
                CGRAPH_ECHO("[RAG] embed batch OK (%zu, dim=%zu)", v.size(), v.empty()?0:v[0].size());
                return v;
            } catch (const std::exception& e) {
                CGRAPH_ECHO("[RAG] embed batch failed: %s, fallback mock", e.what());
            }
        }
        std::vector<std::vector<float>> r;
        for (auto& t : texts) r.push_back(mock_embed(t));
        return r;
    }
};

class EmbedMergeNode : public GNode {
public:
    EmbedMergeNode() { this->setName("EmbedMerge"); }
    void setBuffer(std::shared_ptr<EmbedIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* ep = this->getGParam<EmbedParam>("embed");
        if (!ep) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(ep) {
            ep->embeddings = std::move(buf_->embeddings);
            ep->embeddings_small = ep->embeddings;
        }
        CGRAPH_ECHO("[RAG] EmbedMerge: committed (dim=%zu)",
            ep->embeddings.empty() ? 0 : ep->embeddings[0].size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<EmbedIntermediate> buf_;
};

// ===== Query Embedding: Compute + Merge =====
class QueryEmbedComputeNode : public GNode {
public:
    QueryEmbedComputeNode() : sq_idx_(-1) { this->setName("QEmbedCompute"); }
    void setSubQueryIndex(int idx) { sq_idx_ = idx; }
    void setBuffer(std::shared_ptr<QEmbedIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* qp = this->getGParam<QueryParam>("query");
        if (!qp) return STATUS_ERR;
        std::string text;
        CGRAPH_PARAM_READ_REGION(qp) {
            text = (sq_idx_ >= 0 && sq_idx_ < (int)qp->sub_queries.size())
                ? qp->sub_queries[sq_idx_] : qp->query;
        }
        buf_->embedding = try_real_embed_single(text);
        CGRAPH_ECHO("[RAG] QEmbedCompute[%d]: done (dim=%zu)", sq_idx_, buf_->embedding.size());
        return STATUS_OK;
    }

private:
    int sq_idx_;
    std::shared_ptr<QEmbedIntermediate> buf_;

    std::vector<float> try_real_embed_single(const std::string& text) {
        if (g_embed_client && g_embed_client->is_configured()) {
            try { return g_embed_client->embed_single(text); }
            catch (const std::exception& e) {
                CGRAPH_ECHO("[RAG] embed single failed: %s, mock", e.what());
            }
        }
        return mock_embed(text);
    }
};

class QueryEmbedMergeNode : public GNode {
public:
    QueryEmbedMergeNode() { this->setName("QEmbedMerge"); }
    void setBuffer(std::shared_ptr<QEmbedIntermediate> b) { buf_ = std::move(b); }

    CSTATUS run() override {
        auto* qep = this->getGParam<QueryEmbedParam>("qembed");
        if (!qep) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(qep) {
            qep->query_embeddings.push_back(std::move(buf_->embedding));
        }
        CGRAPH_ECHO("[RAG] QEmbedMerge: committed (total=%zu)",
            qep->query_embeddings.size());
        return STATUS_OK;
    }

private:
    std::shared_ptr<QEmbedIntermediate> buf_;
};

#endif
