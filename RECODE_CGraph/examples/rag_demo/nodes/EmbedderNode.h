#ifndef RAG_EMBEDDER_NODE_H
#define RAG_EMBEDDER_NODE_H

#include "../RAGCommon.h"
#include <EmbeddingClient.h>
#include <random>
#include <memory>

class EmbedderNode : public GNode {
public:
    EmbedderNode() : is_query_mode_(false), sub_query_index_(-1) { this->setName("Embedder"); }
    EmbedderNode(bool query_mode, int sq_idx = -1)
        : is_query_mode_(query_mode), sub_query_index_(sq_idx) {
        this->setName(query_mode ? "QueryEmbed" : "DocEmbed");
    }

    void setQueryMode(bool mode) { is_query_mode_ = mode; this->setName(mode ? "QueryEmbed" : "DocEmbed"); }
    void setSubQueryIndex(int idx) { sub_query_index_ = idx; }

    static void setEmbeddingClient(std::shared_ptr<EmbeddingClient> c) { s_client_ = std::move(c); }
    static bool hasRealEmbedding() { return s_client_ && s_client_->is_configured(); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        if (is_query_mode_) {
            {  // FIX: scope READ lock
            CGRAPH_PARAM_READ_REGION(p) {
                std::string text = (sub_query_index_ >= 0 && sub_query_index_ < (int)p->sub_queries.size())
                    ? p->sub_queries[sub_query_index_] : p->query;
                vcache_ = try_real_embed_single(text);
            }
            }  // FIX: READ lock released
            CGRAPH_PARAM_WRITE_REGION(p) {
                p->query_embeddings.push_back(std::move(vcache_));
            }
            CGRAPH_ECHO("[RAG] %s: done (dim=%zu)",
                this->getName().c_str(), p->query_embeddings.back().size());
        } else {
            {  // FIX: scope READ lock
            CGRAPH_PARAM_READ_REGION(p) {
                auto& src = p->chunks_small.empty() ? p->chunks : p->chunks_small;
                bcache_ = try_real_embed_batch(src);
            }
            }  // FIX: READ lock released
            CGRAPH_PARAM_WRITE_REGION(p) {
                p->embeddings = std::move(bcache_);
                p->embeddings_small = p->embeddings;
            }
            CGRAPH_ECHO("[RAG] DocEmbed: %zu vectors (dim=%zu)",
                p->embeddings.size(),
                p->embeddings.empty() ? 0 : p->embeddings[0].size());
        }
        return STATUS_OK;
    }

private:
    bool is_query_mode_;
    int  sub_query_index_;
    std::vector<float> vcache_;
    std::vector<std::vector<float>> bcache_;

    std::vector<float> try_real_embed_single(const std::string& text) {
        if (s_client_ && s_client_->is_configured()) {
            try {
                auto v = s_client_->embed_single(text);
                CGRAPH_ECHO("[RAG] embed API: OK (dim=%zu)", v.size());
                return v;
            } catch (const std::exception& e) {
                CGRAPH_ECHO("[RAG] embed API failed: %s -- fallback mock", e.what());
            }
        }
        return mock_embed(text);
    }

    std::vector<std::vector<float>> try_real_embed_batch(const std::vector<std::string>& texts) {
        if (texts.empty()) return {};
        if (s_client_ && s_client_->is_configured()) {
            try {
                auto v = s_client_->embed(texts);
                CGRAPH_ECHO("[RAG] embed batch: OK (%zu texts, dim=%zu)",
                    v.size(), v.empty() ? 0 : v[0].size());
                return v;
            } catch (const std::exception& e) {
                CGRAPH_ECHO("[RAG] embed batch failed: %s -- fallback mock", e.what());
            }
        }
        std::vector<std::vector<float>> r;
        for (auto& t : texts) r.push_back(mock_embed(t));
        return r;
    }

    static std::vector<float> mock_embed(const std::string& text) {
        std::vector<float> vec(128);
        std::mt19937 rng((unsigned)std::hash<std::string>{}(text));
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float norm = 0.0f;
        for (auto& v : vec) { v = dist(rng); norm += v * v; }
        norm = std::sqrt(norm);
        if (norm > 0) for (auto& v : vec) v /= norm;
        return vec;
    }

    inline static std::shared_ptr<EmbeddingClient> s_client_;
};

#endif