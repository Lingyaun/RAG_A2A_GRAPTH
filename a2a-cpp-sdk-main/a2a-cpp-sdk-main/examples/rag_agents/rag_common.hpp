#pragma once
// RAG Agent 间通信的消息格式定义
// 基于 A2A JSON-RPC 2.0 协议

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ---- 检索请求 ----
struct RetrieveRequest {
    std::string query;
    int top_k = 10;

    json to_json() const {
        return {{"query", query}, {"top_k", top_k}};
    }
    static RetrieveRequest from_json(const json& j) {
        RetrieveRequest r;
        r.query = j.value("query", "");
        r.top_k = j.value("top_k", 10);
        return r;
    }
};

// ---- 检索结果 ----
struct RetrieveResult {
    int chunk_index;
    float score;
    std::string text;

    json to_json() const {
        return {{"chunk_index", chunk_index}, {"score", score}, {"text", text}};
    }
};

struct RetrieveResponse {
    std::vector<RetrieveResult> results;

    json to_json() const {
        json arr = json::array();
        for (auto& r : results) arr.push_back(r.to_json());
        return {{"results", arr}};
    }
};

// ---- 生成请求 ----
struct GenerateRequest {
    std::string query;
    std::string context;   // 拼接好的检索结果

    json to_json() const {
        return {{"query", query}, {"context", context}};
    }
    static GenerateRequest from_json(const json& j) {
        GenerateRequest g;
        g.query = j.value("query", "");
        g.context = j.value("context", "");
        return g;
    }
};

// ---- 生成响应 ----
struct GenerateResponse {
    std::string answer;

    json to_json() const {
        return {{"answer", answer}};
    }
    static GenerateResponse from_json(const json& j) {
        GenerateResponse g;
        g.answer = j.value("answer", "");
        return g;
    }
};
