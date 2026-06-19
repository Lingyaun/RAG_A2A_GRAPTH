// ================================================================
// retriever_agent.cpp
// RAG Retriever Agent: 独立的 HTTP 服务，封装 CGraph 向量检索
//
// A2A 端点:
//   POST /                 → JSON-RPC 2.0 消息处理
//   GET  /.well-known/agent-card.json → Agent 能力描述
//
// 内部:
//   使用 CGraph 的 VectorSearchNode + FusionRerankerNode
//   对同一个查询并行搜索多个分片
// ================================================================

#include "../multi_agent_demo/http_server.hpp"
#include "../multi_agent_demo/registry_client.hpp"
#include "../multi_agent_demo/qwen_client.hpp"
#include "rag_common.hpp"

// 引入 CGraph 框架
#include "../../../../RECODE_CGraph/src/GraphCtrl/GraphInclude.h"

// ---- 内嵌 CGraph RAG 节点 ----
// 简化版：Agent 内部不使用文件加载，直接从注册时传入的知识库参数中获取数据
// 实际部署时这些数据来自 Redis 或本地文件

// 余弦相似度（内嵌版本，避免依赖 RAGCommon.h）
static std::vector<std::vector<float>> g_embeddings;
static std::vector<std::string> g_chunks;

static float cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
    float dot=0,na=0,nb=0;
    for(size_t i=0;i<a.size();++i){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return (na&&nb)?dot/(std::sqrt(na)*std::sqrt(nb)):0.0f;
}

// ---- A2A 服务主逻辑 ----
class RetrieverAgent {
    std::string agent_id_;
    int port_;
    std::shared_ptr<RegistryClient> registry_;

public:
    RetrieverAgent(const std::string& id, int port,
                   const std::string& registry_url,
                   const std::vector<std::vector<float>>& embeddings,
                   const std::vector<std::string>& chunks)
        : agent_id_(id), port_(port) {
        g_embeddings = embeddings;
        g_chunks = chunks;
        registry_ = std::make_shared<RegistryClient>(registry_url);
    }

    void start() {
        HttpServer server(port_);

        // JSON-RPC 端点
        server.register_handler("/", [this](const std::string& body) {
            return handle_jsonrpc(body);
        });

        // Agent Card 端点
        server.register_handler("/.well-known/agent-card.json", [this](const std::string&) {
            json card = {
                {"name", "RAG Retriever"},
                {"description", "Vector search for RAG pipeline"},
                {"id", agent_id_},
                {"capabilities", {{"retrieve", true}}},
                {"tags", {"retriever", "rag", "search"}}
            };
            return card.dump();
        });

        // 启动服务
        std::thread server_thread([&server](){ server.start(); });
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 注册到服务中心
        AgentRegistration reg;
        reg.id = agent_id_;
        reg.name = "Retriever-" + agent_id_;
        reg.address = "http://localhost:" + std::to_string(port_);
        reg.tags = {"retriever", "rag"};

        if (registry_->register_agent(reg)) {
            std::cout << "[Retriever:" << agent_id_ << "] registered" << std::endl;
        }

        server_thread.join();
    }

private:
    std::string handle_jsonrpc(const std::string& body) {
        try {
            auto req = json::parse(body);
            std::string method = req.value("method", "");

            if (method == "retrieve") {
                auto rr = RetrieveRequest::from_json(req["params"]);
                auto resp = do_retrieve(rr);

                json result;
                result["jsonrpc"] = "2.0";
                result["id"] = req.value("id", json(nullptr));
                result["result"] = resp.to_json();
                return result.dump();
            }

            // 错误响应
            json err;
            err["jsonrpc"] = "2.0";
            err["id"] = req.value("id", json(nullptr));
            err["error"] = {{"code", -32601}, {"message", "Method not found"}};
            return err.dump();

        } catch (const std::exception& e) {
            json err;
            err["jsonrpc"] = "2.0";
            err["id"] = json(nullptr);
            err["error"] = {{"code", -32700}, {"message", e.what()}};
            return err.dump();
        }
    }

    RetrieveResponse do_retrieve(const RetrieveRequest& req) {
        // 使用 CGraph 并行搜索
        // 实际实现：注册 VectorSearchNode × 2 (多分片) → FusionRerankerNode
        // 此处为简化版，直接计算
        std::vector<std::pair<int, float>> scores;
        for (size_t i = 0; i < g_embeddings.size(); ++i) {
            // 模拟查询向量（实际应由 Embedding API 生成）
            std::vector<float> qv(128, 0.1f);
            float s = cos_sim(qv, g_embeddings[i]);
            scores.push_back({(int)i, s});
        }
        std::sort(scores.begin(), scores.end(),
                  [](auto& a, auto& b){ return a.second > b.second; });

        int k = std::min(req.top_k, (int)scores.size());
        RetrieveResponse resp;
        for (int i = 0; i < k; ++i) {
            RetrieveResult r;
            r.chunk_index = scores[i].first;
            r.score = scores[i].second;
            r.text = g_chunks[scores[i].first].substr(0, 200);
            resp.results.push_back(r);
        }

        std::cout << "[Retriever:" << agent_id_
                  << "] query='" << req.query.substr(0,50)
                  << "' top=" << k << std::endl;
        return resp;
    }
};

// ---- Main ----
int main(int argc, char* argv[]) {
    std::string agent_id = "retriever-1";
    int port = 9001;
    std::string registry_url = "http://localhost:8500";

    if (argc > 1) agent_id = argv[1];
    if (argc > 2) port = std::stoi(argv[2]);

    // 模拟知识库（实际部署时从文件/Redis加载）
    std::vector<std::string> chunks = {
        "C++11 introduced std::thread, std::mutex, and std::atomic.",
        "C++17 added Parallel STL and std::shared_mutex.",
        "Work-stealing thread pools improve CPU utilization.",
        "Lock-free data structures reduce contention overhead."
    };
    std::vector<std::vector<float>> embeddings(chunks.size(), std::vector<float>(128, 0.0f));
    // 简单模拟向量
    for (size_t i = 0; i < chunks.size(); ++i)
        embeddings[i][(int)i % 128] = 1.0f;

    std::cout << "[Retriever:" << agent_id << "] starting on port " << port << std::endl;

    RetrieverAgent agent(agent_id, port, registry_url, embeddings, chunks);
    agent.start();
    return 0;
}
