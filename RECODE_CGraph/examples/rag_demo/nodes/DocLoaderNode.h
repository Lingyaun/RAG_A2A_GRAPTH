#ifndef RAG_DOCLOADER_NODE_H
#define RAG_DOCLOADER_NODE_H

#include "../RAGCommon.h"
#include <fstream>
#include <sstream>

// ============================================================
// DocLoaderNode - 文档加载节点
// 使用方式：注册后通过 setPaths() 设置文件列表
// ============================================================
class DocLoaderNode : public GNode {
public:
    DocLoaderNode() { this->setName("DocLoader"); }

    void setPaths(const std::vector<std::string>& p) { paths_ = p; }

    CSTATUS init() override {
        return this->createGParam<RAGParam>("rag");
    }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) { CGRAPH_ECHO("[RAG] DocLoader: RAGParam not found"); return STATUS_ERR; }
        CGRAPH_PARAM_WRITE_REGION(p) {
            for (const auto& path : paths_) {
                std::ifstream file(path);
                if (!file.is_open()) {
                    CGRAPH_ECHO("[RAG] DocLoader: cannot open %s", path.c_str());
                    continue;
                }
                std::stringstream buf; buf << file.rdbuf();
                p->documents.push_back(buf.str());
            }
        }
        CGRAPH_ECHO("[RAG] DocLoader: loaded %zu docs", p->documents.size());
        return STATUS_OK;
    }
private:
    std::vector<std::string> paths_;
};

#endif
