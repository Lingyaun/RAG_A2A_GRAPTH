#ifndef RAG_LLM_GENERATOR_NODE_H
#define RAG_LLM_GENERATOR_NODE_H

#include "../RAGCommon.h"
#include <sstream>
#include <map>

class LLMGeneratorNode : public GNode {
public:
    LLMGeneratorNode() { this->setName("Generator"); }

    CSTATUS run() override {
        auto* pp = this->getGParam<ParentParam>("parent");
        auto* fp = this->getGParam<FusionParam>("fusion");
        auto* dp = this->getGParam<DocParam>("doc");
        auto* qp = this->getGParam<QueryParam>("query");
        if (!pp || !fp || !dp || !qp) return STATUS_ERR;

        std::vector<std::pair<int, float>> selected;
        std::vector<std::string> src_chunks;
        std::string chunk_type;
        bool found = false;

        // Priority 1: parent_chunks
        {
            CGRAPH_PARAM_READ_REGION(pp) {
                if (!pp->parent_chunks.empty()) {
                    selected = pp->parent_chunks;
                    found = true;
                }
            }
        }
        if (found) {
            CGRAPH_PARAM_READ_REGION(dp) { src_chunks = dp->chunks_large; }
            chunk_type = "parent_paragraphs";
        }

        // Priority 2: fused_results
        if (!found) {
            CGRAPH_PARAM_READ_REGION(fp) {
                if (!fp->fused_results.empty()) {
                    selected = fp->fused_results;
                    found = true;
                }
            }
            if (found) {
                CGRAPH_PARAM_READ_REGION(dp) { src_chunks = dp->chunks_small.empty() ? dp->chunks : dp->chunks_small; }
                chunk_type = "small_chunks";
            }
        }

        // Priority 3: fallback merge from SearchParam
        if (!found) {
            auto* sp = this->getGParam<SearchParam>("search");
            if (sp) {
                CGRAPH_PARAM_READ_REGION(sp) {
                    std::map<int, float> merged;
                    for (auto& track : sp->top_k)
                        for (auto& kv : track)
                            if (!merged.count(kv.first) || kv.second > merged[kv.first])
                                merged[kv.first] = kv.second;
                    for (auto& kv : merged) selected.push_back(kv);
                }
            }
            std::sort(selected.begin(), selected.end(), [](auto& a, auto& b) { return a.second > b.second; });
            if (selected.size() > 5) selected.resize(5);
            CGRAPH_PARAM_READ_REGION(dp) { src_chunks = dp->chunks_small.empty() ? dp->chunks : dp->chunks_small; }
            chunk_type = "fallback_merged";
        }

        std::ostringstream ctx;
        std::string q;
        {
            CGRAPH_PARAM_READ_REGION(qp) { q = qp->query; }
            for (size_t i = 0; i < selected.size(); ++i) {
                int idx = selected[i].first;
                if (idx >= 0 && idx < (int)src_chunks.size()) {
                    ctx << "[" << chunk_type << " " << idx
                        << " | score=" << selected[i].second << "]\n"
                        << src_chunks[idx] << "\n\n";
                }
            }
        }

        std::string prompt =
            "Answer based ONLY on the materials below.\n\n"
            "=== Materials ===\n" + ctx.str() +
            "=== End Materials ===\n\n"
            "Question: " + q + "\n\nAnswer:";

        auto* ap = this->getGParam<AnswerParam>("answer");
        CGRAPH_PARAM_WRITE_REGION(ap) { ap->answer = prompt; }
        CGRAPH_ECHO("[RAG] Generator: prompt %zu chars, %zu %s",
                    prompt.size(), selected.size(), chunk_type.c_str());
        return STATUS_OK;
    }
};

#endif