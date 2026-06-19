#ifndef RAG_CHUNKER_NODE_H
#define RAG_CHUNKER_NODE_H

#include "../RAGCommon.h"

// ============================================================
// ChunkerNode - 滑动窗口文本切分
// ============================================================
class ChunkerNode : public GNode {
public:
    ChunkerNode() : chunk_size_(512), overlap_(50) { this->setName("Chunker"); }
    ChunkerNode(int cs, int ol) : chunk_size_(cs), overlap_(ol) { this->setName("Chunker"); }

    CSTATUS run() override {
        auto* p = this->getGParam<RAGParam>("rag");
        if (!p) return STATUS_ERR;
        CGRAPH_PARAM_WRITE_REGION(p) {
            for (const auto& doc : p->documents) {
                if (doc.empty()) continue;
                int total = (int)doc.size(), start = 0, step = chunk_size_ - overlap_;
                if (step <= 0) step = chunk_size_;
                while (start < total) {
                    int end = start + chunk_size_;
                    if (end > total) end = total;
                    p->chunks.push_back(doc.substr(start, end - start));
                    if (end >= total) break;
                    start += step;
                }
            }
        }
        CGRAPH_ECHO("[RAG] Chunker: %zu chunks (size=%d overlap=%d)",
                    p->chunks.size(), chunk_size_, overlap_);
        return STATUS_OK;
    }
private:
    int chunk_size_, overlap_;
};

#endif
