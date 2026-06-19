#ifndef RAG_FUSION_RERANKER_NODE_H
#define RAG_FUSION_RERANKER_NODE_H

#include "../RAGCommon.h"
#include <map>

class FusionRerankerNode : public GNode {
    int final_top_k_ = 5;
public:
    FusionRerankerNode() { this->setName("FusionReranker"); }
    FusionRerankerNode(int k) : final_top_k_(k) { this->setName("FusionReranker"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        std::map<int, float> merged;
        int total_input = 0;

        {
            CGRAPH_PARAM_READ_REGION(p) {
                for (auto& track : p->top_k) {
                    total_input += (int)track.size();
                    for (auto& kv : track) {
                        int idx = kv.first;
                        float score = kv.second;
                        if (!merged.count(idx) || score > merged[idx])
                            merged[idx] = score;
                    }
                }
            }
        }

        std::vector<std::pair<int, float>> ranked;
        for (auto& kv : merged) ranked.push_back(kv);
        std::sort(ranked.begin(), ranked.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        if ((int)ranked.size() > final_top_k_) ranked.resize(final_top_k_);

        CGRAPH_PARAM_WRITE_REGION(p) { p->fused_results = ranked; }

        CGRAPH_ECHO("[RAG] [%s] %d items from %zu tracks -> %zu unique, top=%zu",
                    this->getName().c_str(), total_input,
                    p->top_k.size(), merged.size(), ranked.size());
        return STATUS_OK;
    }
};

#endif
