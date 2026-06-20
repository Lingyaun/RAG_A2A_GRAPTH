#ifndef RAG_QUERY_DECOMPOSER_NODE_H
#define RAG_QUERY_DECOMPOSER_NODE_H

#include "../RAGCommon.h"

class QueryDecomposerNode : public GNode {
public:
    QueryDecomposerNode() { this->setName("Decomposer"); }

    CSTATUS run() override {
        auto* qp = this->getGParam<QueryParam>("query");
        if (!qp) return STATUS_ERR;

        CGRAPH_PARAM_WRITE_REGION(qp) {
            auto parts = split_by_string(qp->query, "?");
            if (parts.empty()) parts.push_back(qp->query);
            if (parts.size() <= 1) {
                qp->sub_queries.push_back(qp->query);
                qp->sub_queries.push_back(qp->query + " details");
                qp->sub_queries.push_back(qp->query + " comparison");
                qp->sub_queries.push_back(qp->query + " examples");
            } else {
                for (auto& part : parts) {
                    if (!part.empty()) qp->sub_queries.push_back(part);
                }
                while (qp->sub_queries.size() < 4) {
                    qp->sub_queries.push_back(qp->query);
                }
            }
            if (qp->sub_queries.size() > 4) {
                qp->sub_queries.resize(4);
            }
        }

        // Read back for logging (no concurrent writer at this stage)
        auto* qp2 = this->getGParam<QueryParam>("query");
        CGRAPH_ECHO("[RAG] Decomposer: %zu sub-queries", qp2->sub_queries.size());
        for (size_t i = 0; i < qp2->sub_queries.size() && i < 4; ++i) {
            CGRAPH_ECHO("[RAG]   SQ%d: %s", (int)i, qp2->sub_queries[i].c_str());
        }
        return STATUS_OK;
    }
};

#endif