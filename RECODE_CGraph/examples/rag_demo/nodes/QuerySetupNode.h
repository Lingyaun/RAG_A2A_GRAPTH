#ifndef RAG_QUERY_SETUP_NODE_H
#define RAG_QUERY_SETUP_NODE_H

#include "../RAGCommon.h"

class QuerySetupNode : public GNode {
public:
    QuerySetupNode() { this->setName("QuerySetup"); }
    void setQuery(const std::string& q) { query_ = q; }

    CSTATUS run() override {
        auto* qp = this->getGParam<QueryParam>("query");
        if (!qp) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(qp) { qp->query = query_; }
        CGRAPH_ECHO("[RAG] QuerySetup: %s", query_.c_str());
        return STATUS_OK;
    }
private:
    std::string query_;
};

#endif