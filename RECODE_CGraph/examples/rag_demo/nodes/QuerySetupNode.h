#ifndef RAG_QUERY_SETUP_NODE_H
#define RAG_QUERY_SETUP_NODE_H

#include "../RAGCommon.h"

// ============================================================
// QuerySetupNode - 创建 RAGParam 并写入查询文本
// 使用方式：注册后通过 setQuery() 设置查询
// ============================================================
class QuerySetupNode : public GNode {
public:
    QuerySetupNode() { this->setName("QuerySetup"); }

    void setQuery(const std::string& q) { query_ = q; }

    CSTATUS init() override {
        return this->createGParam<RAGParam>("rag");
    }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(p) { p->query = query_; }
        CGRAPH_ECHO("[RAG] QuerySetup: %s", query_.c_str());
        return STATUS_OK;
    }
private:
    std::string query_;
};

#endif
