#ifndef RAG_PARENT_LOOKUP_NODE_H
#define RAG_PARENT_LOOKUP_NODE_H

#include "../RAGCommon.h"

class ParentLookupNode : public GNode {
public:
    ParentLookupNode() { this->setName("ParentLookup"); }

    CSTATUS run() override {
        auto* fp = this->getGParam<FusionParam>("fusion");
        auto* dp = this->getGParam<DocParam>("doc");
        if (!fp || !dp) return STATUS_ERR;

        std::vector<std::pair<int, float>> fused;
        { CGRAPH_PARAM_READ_REGION(fp) { fused = fp->fused_results; }}

        std::unordered_map<int, float> parent_scores;
        { CGRAPH_PARAM_READ_REGION(dp) {
            if (dp->small_to_parent.empty()) {
                // Degrade: no mapping table -> passthrough
                auto* pp = this->getGParam<ParentParam>("parent");
                CGRAPH_PARAM_WRITE_REGION(pp) { pp->parent_chunks = fused; }
                CGRAPH_ECHO("[RAG] ParentLookup: no map, passthrough %zu chunks", fused.size());
                return STATUS_OK;
            }
            for (auto& [small_id, score] : fused) {
                if (small_id < 0 || small_id >= (int)dp->small_to_parent.size()) continue;
                int parent_id = dp->small_to_parent[small_id];
                auto it = parent_scores.find(parent_id);
                if (it == parent_scores.end() || score > it->second)
                    parent_scores[parent_id] = score;
            }
        }}

        std::vector<std::pair<int, float>> results;
        for (auto& [pid, score] : parent_scores) results.push_back({pid, score});
        std::sort(results.begin(), results.end(), [](auto& a, auto& b) { return a.second > b.second; });

        auto* pp = this->getGParam<ParentParam>("parent");
        CGRAPH_PARAM_WRITE_REGION(pp) { pp->parent_chunks = results; }

        CGRAPH_ECHO("[RAG] ParentLookup: %zu small -> %zu parents", fused.size(), results.size());
        return STATUS_OK;
    }
};

#endif