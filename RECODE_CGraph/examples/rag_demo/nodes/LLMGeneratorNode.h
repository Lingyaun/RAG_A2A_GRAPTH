#ifndef RAG_LLM_GENERATOR_NODE_H
#define RAG_LLM_GENERATOR_NODE_H

#include "../RAGCommon.h"
#include <sstream>
#include <map>

// ============================================================
// LLMGeneratorNode - 拼接 Context + 构造 Prompt
// Phase1: 存 prompt 文本; Phase2: 替换为 LLM API 调用
// ============================================================
class LLMGeneratorNode : public GNode {
public:
    LLMGeneratorNode() { this->setName("Generator"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        std::vector<std::pair<int, float>> selected;
        {
            CGRAPH_PARAM_READ_REGION(p) {
                if (!p->fused_results.empty()) {
                    selected = p->fused_results;
                } else {
                    std::map<int, float> merged;
                    for (auto& track : p->top_k) {
                        for (auto& kv : track) {
                            if (!merged.count(kv.first) || kv.second > merged[kv.first])
                                merged[kv.first] = kv.second;
                        }
                    }
                    for (auto& kv : merged) selected.push_back(kv);
                    std::sort(selected.begin(), selected.end(),
                        [](auto& a, auto& b) { return a.second > b.second; });
                    if (selected.size() > 5) selected.resize(5);
                }
            }
        }

        std::ostringstream ctx;
        std::string q;
        {
            CGRAPH_PARAM_READ_REGION(p) {
                for (size_t i = 0; i < selected.size(); ++i) {
                    int idx = selected[i].first;
                    if (idx >= 0 && idx < (int)p->chunks.size()) {
                        ctx << "[Chunk" << idx << "|score="
                            << selected[i].second << "]\n"
                            << p->chunks[idx] << "\n\n";
                    }
                }
                q = p->query;
            }
        }

        std::string prompt =
            "Answer based ONLY on the materials below.\n\n"
            "=== Materials ===\n" + ctx.str() +
            "=== End Materials ===\n\n"
            "Question: " + q + "\n\nAnswer:";

        CGRAPH_PARAM_WRITE_REGION(p) { p->answer = prompt; }
        CGRAPH_ECHO("[RAG] Generator: prompt %zu chars, %zu chunks",
                    prompt.size(), selected.size());
        return STATUS_OK;
    }
};

#endif
