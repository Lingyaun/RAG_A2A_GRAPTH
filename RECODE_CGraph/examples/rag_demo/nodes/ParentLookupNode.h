#ifndef RAG_PARENT_LOOKUP_NODE_H
#define RAG_PARENT_LOOKUP_NODE_H

#include "../RAGCommon.h"

// ============================================================
// ParentLookupNode — 层次索引核心：小chunk → 父段落映射
//
// 输入: RAGParam::fused_results (small chunk IDs)
// 输出: RAGParam::parent_chunks (去重后的父段落 IDs)
//
// 关键设计:
//   - 多个 small chunk 可能映射到同一 parent → 取最高分
//   - 索引越界保护: small_id < small_to_parent.size()
//   - O(候选数) 时间复杂度，零额外 IO
// ============================================================
class ParentLookupNode : public GNode {
public:
    ParentLookupNode() { this->setName("ParentLookup"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        // 1. 读取融合结果
        std::vector<std::pair<int, float>> fused;
        {
            CGRAPH_PARAM_READ_REGION(p) { fused = p->fused_results; }
        }

        // 2. 查映射表，收集父段落
        std::unordered_map<int, float> parent_scores;

        {
            CGRAPH_PARAM_READ_REGION(p) {
                if (p->small_to_parent.empty()) {
                // 降级：无映射表 → 直接透传（兼容旧 Chunker）
                    CGRAPH_PARAM_WRITE_REGION(p) { p->parent_chunks = fused; }
                    CGRAPH_ECHO("[RAG] ParentLookup: no map, passthrough %zu chunks",
                                fused.size());
                    return STATUS_OK;
                }

                for (auto& [small_id, score] : fused) {
                    if (small_id < 0 || small_id >= (int)p->small_to_parent.size())
                        continue;

                    int parent_id = p->small_to_parent[small_id];

                    // 多个 small chunk 命中同一 parent → 保留最高分
                    auto it = parent_scores.find(parent_id);
                    if (it == parent_scores.end() || score > it->second)
                        parent_scores[parent_id] = score;
                }
            }
        }

        // 3. 转回排序结果
        std::vector<std::pair<int, float>> results;
        for (auto& [pid, score] : parent_scores)
            results.push_back({pid, score});
        std::sort(results.begin(), results.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        // 4. 写入
        CGRAPH_PARAM_WRITE_REGION(p) { p->parent_chunks = results; }

        CGRAPH_ECHO("[RAG] ParentLookup: %zu small → %zu parents",
                    fused.size(), results.size());
        return STATUS_OK;
    }
};

#endif
