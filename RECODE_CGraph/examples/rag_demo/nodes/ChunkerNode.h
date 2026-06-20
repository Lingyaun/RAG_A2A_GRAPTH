#ifndef RAG_CHUNKER_NODE_H
#define RAG_CHUNKER_NODE_H

#include "../RAGCommon.h"
#include <algorithm>

class ChunkerNode : public GNode {
public:
    int min_chunk_size_ = 128;
    int max_chunk_size_ = 512;
    int overlap_        = 50;

    static inline const std::vector<std::string> SEPARATORS = {
        "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
    };

    ChunkerNode() { this->setName("Chunker"); }

    CSTATUS run() override {
        auto* doc = this->getGParam<DocParam>("doc");
        if (!doc) return STATUS_ERR;

        int total_small = 0, total_large = 0;
        CGRAPH_PARAM_WRITE_REGION(doc) {
            for (const auto& d : doc->documents) {
                if (d.empty()) continue;
                auto paragraphs = split_by_string(d, "\n\n");
                for (const auto& para : paragraphs) {
                    std::string trimmed = para;
                    trim_inplace(trimmed);
                    if (trimmed.empty()) continue;
                    int parent_id = (int)doc->chunks_large.size();
                    doc->chunks_large.push_back(trimmed);
                    ++total_large;
                    auto raw_parts = recursive_split(trimmed, 1);
                    auto merged = merge_short(raw_parts);
                    for (const auto& sp : merged) {
                        doc->chunks_small.push_back(sp);
                        doc->small_to_parent.push_back(parent_id);
                        ++total_small;
                    }
                }
            }
            doc->chunks = doc->chunks_small;
        }
        CGRAPH_ECHO("[RAG] Chunker: %d paragraphs -> %d small / %d large",
                    total_large, total_small, total_large);
        return STATUS_OK;
    }

private:
    std::vector<std::string> recursive_split(const std::string& text, int sep_level) {
        if (sep_level >= (int)SEPARATORS.size()) {
            std::vector<std::string> result;
            for (size_t i = 0; i < text.size(); i += (size_t)max_chunk_size_) {
                std::string piece = text.substr(i, max_chunk_size_);
                trim_inplace(piece);
                if (!piece.empty()) result.push_back(piece);
            }
            return result;
        }
        auto parts = split_by_string(text, SEPARATORS[sep_level]);
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
                auto sub = recursive_split(part, sep_level + 1);
                result.insert(result.end(), sub.begin(), sub.end());
            }
        }
        return result;
    }

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
        if (result.empty() && !parts.empty()) return parts;
        return result;
    }
};

#endif