#ifndef RAG_DOCLOADER_NODE_H
#define RAG_DOCLOADER_NODE_H

#include "../RAGCommon.h"
#include <fstream>
#include <sstream>

class DocLoaderNode : public GNode {
public:
    DocLoaderNode() { this->setName("DocLoader"); }
    void setPaths(const std::vector<std::string>& p) { paths_ = p; }

    CSTATUS run() override {
        auto* doc = this->getGParam<DocParam>("doc");
        if (!doc) { CGRAPH_ECHO("[RAG] DocLoader: DocParam not found"); return STATUS_ERR; }
        CGRAPH_PARAM_WRITE_REGION(doc) {
            for (const auto& path : paths_) {
                std::ifstream file(path);
                if (!file.is_open()) {
                    CGRAPH_ECHO("[RAG] DocLoader: cannot open %s", path.c_str());
                    continue;
                }
                std::stringstream buf; buf << file.rdbuf();
                doc->documents.push_back(buf.str());
            }
        }
        CGRAPH_ECHO("[RAG] DocLoader: loaded %zu docs", doc->documents.size());
        return STATUS_OK;
    }

    CSTATUS init() override {
        // DocParam 由 InitNode 创建
        return STATUS_OK;
    }

private:
    std::vector<std::string> paths_;
};

#endif