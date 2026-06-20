#ifndef RAG_CHUNKER_NODE_H
#define RAG_CHUNKER_NODE_H

#include "../RAGCommon.h"
#include <algorithm>

// ============================================================
// ChunkerNode — 递归语义切分 + 双粒度输出
//
// 建库时产出 chunks_small (索引用) 和 chunks_large (LLM用)
// 同时构建 small_to_parent 映射表，供 Phase 6 ParentLookupNode 使用
//
// 分隔符优先级：段落 → 句子 → 分句 → 空格 → 字符
// ============================================================
class ChunkerNode : public GNode {
public:
    // ---- 配置参数 ----
    int min_chunk_size_ = 128;   // 小于此合并到前一个（小chunk最小粒度）
    int max_chunk_size_ = 512;   // 大于此强制按低优先级切分
    int overlap_        = 50;    // chunk 间重叠字符数

    // 递归切分用的分隔符优先级列表
    static inline const std::vector<std::string> SEPARATORS = {
        "\n\n",     // 0: 段落边界（最高优先级）
        "\n",       // 1: 换行
        ". ",       // 2: 英文句号+空格
        "! ",       // 3: 英文感叹号
        "? ",       // 4: 英文问号
        "; ",       // 5: 英文分号
        ", ",       // 6: 逗号
        " ",        // 7: 空格（最后手段）
        ""          // 8: 字符级（最终降级）
    };

    ChunkerNode() { this->setName("Chunker"); }
    ChunkerNode(int min_sz, int max_sz, int ol)
        : min_chunk_size_(min_sz), max_chunk_size_(max_sz), overlap_(ol) {
        this->setName("Chunker");
    }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        int total_small = 0, total_large = 0;

        CGRAPH_PARAM_WRITE_REGION(p) {
            for (const auto& doc : p->documents) {
                if (doc.empty()) continue;

                // ====== Step 1: 按段落分割（最高优先级） ======
                auto paragraphs = split_by_string(doc, "\n\n");

                for (const auto& para : paragraphs) {
                    std::string trimmed = para;
                    trim_inplace(trimmed);
                    if (trimmed.empty()) continue;

                    // ====== Step 2: 父段落作为 large chunk ======
                    int parent_id = (int)p->chunks_large.size();
                    p->chunks_large.push_back(trimmed);
                    ++total_large;

                    // ====== Step 3: 递归切分为 small chunks ======
                    auto raw_parts = recursive_split(trimmed, 1); // 从 level 1 开始（跳过段落级）

                    // 合并过短的 chunk
                    auto merged = merge_short(raw_parts);

                    // ====== Step 4: 记录映射关系 ======
                    for (const auto& sp : merged) {
                        p->chunks_small.push_back(sp);
                        p->small_to_parent.push_back(parent_id);
                        ++total_small;
                    }
                }
            }

            // 兼容：让旧代码仍然可以工作
            p->chunks = p->chunks_small;
        }

        CGRAPH_ECHO("[RAG] Chunker: %d paragraphs → %d small chunks (128-%d) / %d large chunks",
                    total_large, total_small, max_chunk_size_, total_large);
        return STATUS_OK;
    }

private:
    // ---- 递归切分 ----
    std::vector<std::string> recursive_split(const std::string& text, int sep_level) {
        if (sep_level >= (int)SEPARATORS.size()) {
            // 最终降级：按字符截断
            std::vector<std::string> result;
            for (size_t i = 0; i < text.size(); i += (size_t)max_chunk_size_) {
                std::string piece = text.substr(i, max_chunk_size_);
                trim_inplace(piece);
                if (!piece.empty()) result.push_back(piece);
            }
            return result;
        }

        auto parts = split_by_string(text, SEPARATORS[sep_level]);

        // 如果当前分隔符切不动（只有一个结果），降级
        if (parts.size() <= 1 && !SEPARATORS[sep_level].empty()) {
            return recursive_split(text, sep_level + 1);
        }

        std::vector<std::string> result;
        for (auto& part : parts) {
            trim_inplace(part);
            if (part.empty()) continue;

            if ((int)part.size() <= max_chunk_size_) {
                result.push_back(part);
            } else {
                // 超长 → 用下一级分隔符递归切分
                auto sub = recursive_split(part, sep_level + 1);
                result.insert(result.end(), sub.begin(), sub.end());
            }
        }
        return result;
    }

    // ---- 合并过短 chunk ----
    std::vector<std::string> merge_short(const std::vector<std::string>& parts) {
        std::vector<std::string> result;
        std::string pending;

        for (const auto& part : parts) {
            if (pending.empty()) {
                pending = part;
                if ((int)pending.size() >= min_chunk_size_) {
                    result.push_back(pending);
                    pending.clear();
                }
            } else {
                std::string merged = pending + " " + part;
                if ((int)merged.size() <= max_chunk_size_) {
                    pending = merged;
                } else {
                    result.push_back(pending);
                    pending = part;
                }
                if ((int)pending.size() >= min_chunk_size_) {
                    result.push_back(pending);
                    pending.clear();
                }
            }
        }
        if (!pending.empty()) result.push_back(pending);

        // 如果合并后一个都没有，至少返回原part
        if (result.empty() && !parts.empty())
            return parts;
        return result;
    }
};

#endif
