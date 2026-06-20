#ifndef RAG_LLM_GENERATOR_NODE_H
#define RAG_LLM_GENERATOR_NODE_H

#include "../RAGCommon.h"
#include <sstream>
#include <map>

// ============================================================
// LLMGeneratorNode - 拼接 Context + 构造 Prompt
//
// Phase 6 扩展：优先使用 parent_chunks（父段落）作为上下文
// 如果 parent_chunks 为空，回退到 fused_results (small chunks)
// 确保 LLM 收到的是完整段落而非碎片化句子
// ============================================================
class LLMGeneratorNode : public GNode {
public:
    LLMGeneratorNode() { this->setName("Generator"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        std::vector<std::pair<int, float>> selected;
        std::vector<std::string> src_chunks;  // 用于获取原文
        std::string chunk_type;

        {
            CGRAPH_PARAM_READ_REGION(p) {
                // Phase 6: 优先使用 parent_chunks（父段落）
                if (!p->parent_chunks.empty()) {
                    selected = p->parent_chunks;
                    src_chunks = p->chunks_large;
                    chunk_type = "parent_paragraphs";
                } else if (!p->fused_results.empty()) {
                    // 回退: 使用 fused_results (small chunks)
                    selected = p->fused_results;
                    src_chunks = p->chunks_small.empty() ? p->chunks : p->chunks_small;
                    chunk_type = "small_chunks";
                } else {
                    // 最终回退: 从 top_k 中合并
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
                    src_chunks = p->chunks_small.empty() ? p->chunks : p->chunks_small;
                    chunk_type = "fallback_merged";
                }
            }
        }

        std::ostringstream ctx;
        std::string q;
        {
            CGRAPH_PARAM_READ_REGION(p) {
                for (size_t i = 0; i < selected.size(); ++i) {
                    int idx = selected[i].first;
                    if (idx >= 0 && idx < (int)src_chunks.size()) {
                        ctx << "[" << chunk_type << " " << idx
                            << " | score=" << selected[i].second << "]\n"
                            << src_chunks[idx] << "\n\n";
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
        CGRAPH_ECHO("[RAG] Generator: prompt %zu chars, %zu %s",
                    prompt.size(), selected.size(), chunk_type.c_str());
        return STATUS_OK;
    }
};

#endif
