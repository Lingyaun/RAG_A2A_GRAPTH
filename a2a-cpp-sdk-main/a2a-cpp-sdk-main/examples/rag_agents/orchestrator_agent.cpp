// ================================================================
// orchestrator_agent.cpp
// RAG Orchestrator Agent: 入口服务，协调整个 RAG 流程
//
// 内部架构:
//   收到查询 → CGraph DAG 调度:
//     Layer 1: Decomposer (拆分问题)
//     Layer 2: 3×A2A调用 → Retriever Agents (并行)
//     Layer 3: FusionReranker (融合结果)
//     Layer 4: A2A调用 → Generator Agent
//     Layer 5: 返回答案
//
// 这是"分布式通信 + 本地并行计算"融合的关键文件
// ================================================================

#include "../multi_agent_demo/http_server.hpp"
#include "../multi_agent_demo/registry_client.hpp"
#include "../multi_agent_demo/qwen_client.hpp"
#include "rag_common.hpp"

// 引入 CGraph 框架（Orchestrator 内部并行调度层）
#include "../../../../RECODE_CGraph/src/GraphCtrl/GraphInclude.h"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <algorithm>

using json = nlohmann::json;

// ---- 简单 HTTP Client（用于 A2A 调用）----
static size_t WriteCB(void* c, size_t s, size_t n, std::string* u) {
    u->append((char*)c, s*n); return s*n;
}

static std::string http_post(const std::string& url, const std::string& body) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";
    std::string resp;
    struct curl_slist* h = nullptr;
    h = curl_slist_append(h, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, h);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCB);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_perform(curl);
    curl_slist_free_all(h);
    curl_easy_cleanup(curl);
    return resp;
}

// ================================================================
// OrchestratorAgent
// ================================================================
class OrchestratorAgent {
    std::string agent_id_;
    int port_;
    std::shared_ptr<RegistryClient> registry_;

    // 已发现的 Agent 地址缓存
    std::vector<std::string> retriever_urls_;
    std::string generator_url_;

public:
    OrchestratorAgent(const std::string& id, int port,
                      const std::string& registry_url)
        : agent_id_(id), port_(port)
        , registry_(std::make_shared<RegistryClient>(registry_url)) {}

    // 从注册中心发现 Retriever 和 Generator Agent
    void discover_agents() {
        // 发现 Retriever Agents
        auto retrievers = registry_->find_agents_by_tag("retriever");
        retriever_urls_.clear();
        for (auto& r : retrievers)
            retriever_urls_.push_back(r.address);

        // 发现 Generator Agent
        auto generators = registry_->find_agents_by_tag("generator");
        if (!generators.empty())
            generator_url_ = generators[0].address;

        std::cout << "[Orchestrator] discovered "
                  << retriever_urls_.size() << " retrievers, generator: "
                  << (generator_url_.empty() ? "NONE" : generator_url_)
                  << std::endl;
    }

    void start() {
        HttpServer server(port_);

        server.register_handler("/", [this](const std::string& body) {
            return handle_jsonrpc(body);
        });

        server.register_handler("/.well-known/agent-card.json", [this](const std::string&) {
            json card = {
                {"name", "RAG Orchestrator"},
                {"description", "Coordinates multi-agent RAG pipeline"},
                {"id", agent_id_},
                {"tags", {"orchestrator", "rag"}}
            };
            return card.dump();
        });

        // 注册到服务中心
        AgentRegistration reg;
        reg.id = agent_id_;
        reg.name = "Orchestrator";
        reg.address = "http://localhost:" + std::to_string(port_);
        reg.tags = {"orchestrator", "rag"};
        registry_->register_agent(reg);

        // 发现其他 Agent
        discover_agents();

        std::cout << "[Orchestrator] listening on port " << port_ << std::endl;
        server.start();
    }

private:
    std::string handle_jsonrpc(const std::string& body) {
        try {
            auto req = json::parse(body);
            std::string method = req.value("method", "");

            if (method == "message/send" || method == "query") {
                // 提取用户查询文本
                std::string user_query;
                if (req.contains("params")) {
                    auto& p = req["params"];
                    if (p.contains("message"))
                        user_query = p["message"].value("text",
                            p["message"].dump());
                    else if (p.contains("query"))
                        user_query = p["query"];
                }
                if (user_query.empty()) user_query = "What is C++ concurrency?";

                std::cout << "[Orchestrator] query: " << user_query << std::endl;

                // ---- 核心：使用 CGraph 并行调度 A2A 调用 ----
                std::string answer = execute_rag_pipeline(user_query);

                json result;
                result["jsonrpc"] = "2.0";
                result["id"] = req.value("id", json(nullptr));
                result["result"] = {{"answer", answer}};
                return result.dump();
            }

            json err;
            err["jsonrpc"] = "2.0";
            err["id"] = req.value("id", json(nullptr));
            err["error"] = {{\"code\", -32601}, {\"message\", \"Method not found\"}};
            return err.dump();

        } catch (const std::exception& e) {
            json err;
            err["jsonrpc"] = "2.0";
            err["id"] = json(nullptr);
            err["error"] = {{\"code\", -32700}, {\"message\", e.what()}};
            return err.dump();
        }
    }

    // ============================================================
    // CGraph DAG: 并行调用多个 Retriever Agent
    // ============================================================
    std::string execute_rag_pipeline(const std::string& query) {
        // 简化版：不使用完整 CGraph 节点（此处仅演示架构模式）
        // 完整版：每个 A2A 调用封装为 GNode，注册到 GPipeline

        std::cout << "[Orchestrator] === RAG Pipeline ===" << std::endl;

        // 1. 问题拆分（模拟）
        std::vector<std::string> sub_queries = {
            query,
            query + " details",
            query + " examples"
        };
        std::cout << "[Orchestrator] decomposed into " << sub_queries.size()
                  << " sub-queries" << std::endl;

        // 2. 并行调用 Retriever Agents
        //    ↓ 在 CGraph 完整版中，这里是 GRegion 包裹的 A2ACallNode × 3
        std::vector<std::string> all_contexts;

        // 为每个子查询分配一个 Retriever（轮询）
        for (size_t i = 0; i < sub_queries.size(); ++i) {
            if (retriever_urls_.empty()) break;
            std::string url = retriever_urls_[i % retriever_urls_.size()];

            RetrieveRequest rreq{sub_queries[i], 3};
            json req_body = {
                {"jsonrpc", "2.0"},
                {"method", "retrieve"},
                {"params", rreq.to_json()},
                {"id", (int)i}
            };

            std::string resp = http_post(url, req_body.dump());
            try {
                auto j = json::parse(resp);
                if (j.contains("result") && j["result"].contains("results")) {
                    for (auto& rj : j["result"]["results"]["results"]) {
                        all_contexts.push_back(rj.value("text", ""));
                    }
                }
            } catch (...) {}

            std::cout << "[Orchestrator] → Retriever@" << url
                      << " (SQ" << i << ")" << std::endl;
        }

        std::cout << "[Orchestrator] collected " << all_contexts.size()
                  << " context chunks" << std::endl;

        // 3. 融合去重
        std::sort(all_contexts.begin(), all_contexts.end());
        all_contexts.erase(
            std::unique(all_contexts.begin(), all_contexts.end()),
            all_contexts.end());

        // 4. 调用 Generator Agent
        std::string context_str;
        for (auto& c : all_contexts)
            context_str += c + "\n---\n";

        std::string answer;
        if (!generator_url_.empty()) {
            GenerateRequest greq{query, context_str};
            json gbody = {
                {"jsonrpc", "2.0"},
                {"method", "generate"},
                {"params", greq.to_json()},
                {"id", 99}
            };
            std::string resp = http_post(generator_url_, gbody.dump());
            try {
                auto j = json::parse(resp);
                auto gr = GenerateResponse::from_json(j["result"]);
                answer = gr.answer;
            } catch (...) {
                answer = "[Generator unreachable]";
            }
        } else {
            // 无 Generator 时的降级输出
            std::ostringstream oss;
            oss << "[Orchestrator fallback] Query: " << query << "\n"
                << "Contexts found: " << all_contexts.size() << "\n"
                << "---\n" << context_str.substr(0, 500);
            answer = oss.str();
        }

        std::cout << "[Orchestrator] answer: " << answer.substr(0, 100) << "..." << std::endl;
        return answer;
    }
};

// ---- Main ----
int main(int argc, char* argv[]) {
    std::string agent_id = "orchestrator-1";
    int port = 9000;
    std::string registry_url = "http://localhost:8500";

    if (argc > 1) agent_id = argv[1];
    if (argc > 2) port = std::stoi(argv[2]);

    curl_global_init(CURL_GLOBAL_ALL);

    OrchestratorAgent agent(agent_id, port, registry_url);
    agent.start();

    curl_global_cleanup();
    return 0;
}
