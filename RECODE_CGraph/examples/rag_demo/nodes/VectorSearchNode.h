#ifndef RAG_VECTOR_SEARCH_NODE_H
#define RAG_VECTOR_SEARCH_NODE_H

#include "../RAGCommon.h"
#include <queue>

class VectorSearchNode : public GNode {
public:
    VectorSearchNode() { this->setName("Search"); }
    VectorSearchNode(int k, int sid, int ns, int qvi)
        : top_k_(k), slice_id_(sid), num_slices_(ns), query_vec_index_(qvi) {
        char buf[64]; snprintf(buf, sizeof(buf), "Search_S%d/%d", sid, ns);
        this->setName(buf);
    }

    void configure(int k, int sid, int ns, int qvi) {
        top_k_ = k; slice_id_ = sid; num_slices_ = ns; query_vec_index_ = qvi;
        char buf[64]; snprintf(buf, sizeof(buf), "Search_S%d/%d", sid, ns);
        this->setName(buf);
    }

    CSTATUS run() override {
        auto* qep = this->getGParam<QueryEmbedParam>("qembed");
        if (!qep) return STATUS_ERR;
        std::vector<float> qv;
        { CGRAPH_PARAM_READ_REGION(qep) {
            if (query_vec_index_ >= (int)qep->query_embeddings.size()) return STATUS_ERR;
            qv = qep->query_embeddings[query_vec_index_];
        }}

        auto* ep = this->getGParam<EmbedParam>("embed");
        if (!ep) return STATUS_ERR;
        size_t total;
        { CGRAPH_PARAM_READ_REGION(ep) { total = ep->embeddings.size(); }}
        if (total == 0) {
            auto* sp = this->getGParam<SearchParam>("search");
            CGRAPH_PARAM_WRITE_REGION(sp) { sp->top_k.push_back({}); }
            return STATUS_OK;
        }

        size_t sz = total / num_slices_;
        size_t s_start = slice_id_ * sz;
        size_t s_end = (slice_id_ == num_slices_ - 1) ? total : (slice_id_ + 1) * sz;

        using SP = std::pair<int, float>;
        auto cmp = [](const SP& a, const SP& b) { return a.second > b.second; };
        std::priority_queue<SP, std::vector<SP>, decltype(cmp)> pq(cmp);

        { CGRAPH_PARAM_READ_REGION(ep) {
            for (size_t i = s_start; i < s_end; ++i) {
                float sim = cosine_similarity(qv, ep->embeddings[i]);
                pq.push({(int)i, sim});
                if ((int)pq.size() > top_k_) pq.pop();
            }
        }}

        std::vector<SP> result;
        while (!pq.empty()) { result.push_back(pq.top()); pq.pop(); }
        std::reverse(result.begin(), result.end());

        auto* sp = this->getGParam<SearchParam>("search");
        CGRAPH_PARAM_WRITE_REGION(sp) { sp->top_k.push_back(result); }
        float best = result.empty() ? 0.0f : result[0].second;
        CGRAPH_ECHO("[RAG] [%s] [%zu,%zu) top1=%.4f",
                    this->getName().c_str(), s_start, s_end, best);
        return STATUS_OK;
    }

private:
    int top_k_ = 10, slice_id_ = 0, num_slices_ = 1, query_vec_index_ = 0;
};

#endif