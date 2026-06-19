#ifndef RAG_QUERY_DECOMPOSER_NODE_H
#define RAG_QUERY_DECOMPOSER_NODE_H

#include "../RAGCommon.h"

// ============================================================
// QueryDecomposerNode - 将复杂问题拆分为子问题
// 策略1（Phase2）：按问号/分号机械拆分
// 策略2（未来）：调LLM分解："把这个问题拆成3个独立子问题"
// ============================================================
class QueryDecomposerNode : public GNode {
public:
    QueryDecomposerNode() { this->setName("Decomposer"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;

        CGRAPH_PARAM_WRITE_REGION(p) {
            // 策略：按主要问号拆分，并自动生成变体查询
            auto parts = split_string(p->query, '?');
            if (parts.size() <= 1) {
                // 单问题 → 生成多个检索角度
                p->sub_queries.push_back(p->query);
                p->sub_queries.push_back(p->query + " details");
                p->sub_queries.push_back(p->query + " comparison");
                p->sub_queries.push_back(p->query + " examples");
            } else {
                for (auto& part : parts) {
                    if (!part.empty()) p->sub_queries.push_back(part);
                }
                // 补全到至少4个
                while (p->sub_queries.size() < 4) {
                    p->sub_queries.push_back(p->query);
                }
            }

            // 限制子问题数量
            if (p->sub_queries.size() > 4) {
                p->sub_queries.resize(4);
            }
        }

        CGRAPH_ECHO("[RAG] Decomposer: %zu sub-queries", p->sub_queries.size());
        for (size_t i = 0; i < p->sub_queries.size(); ++i) {
            CGRAPH_ECHO("[RAG]   SQ%d: %s", (int)i, p->sub_queries[i].c_str());
        }
        return STATUS_OK;
    }
};

#endif
