// ================================================================
// generator_agent.cpp
// RAG Generator Agent: 独立的 HTTP 服务，封装 LLM 调用
//
// 接收: query + context (检索结果拼接的文本)
// 返回: answer (LLM 生成的回答)
// ================================================================

#include "../multi_agent_demo/http_server.hpp"
#include "../multi_agent_demo/registry_client.hpp"
#include "../multi_agent_demo/qwen_client.hpp"
#include "rag_common.hpp"

class GeneratorAgent {
    std::string agent_id_;
    int port_;
    std::shared_ptr<QwenClient> llm_;
    std::shared_ptr<RegistryClient> registry_;

public:
    GeneratorAgent(const std::string& id, int port,
                   const std::string& registry_url,
                   const std::string& api_key)
        : agent_id_(id), port_(port) {
        llm_ = std::make_shared<QwenClient>(api_key);
        registry_ = std::make_shared<RegistryClient>(registry_url);
    }

    void start() {
        HttpServer server(port_);

        server.register_handler("/", [this](const std::string& body) {
            return handle_jsonrpc(body);
        });

        server.register_handler("/.well-known/agent-card.json", [this](const std::string&) {
            json card = {
                {"name", "RAG Generator"},
                {"description", "LLM answer generation for RAG pipeline"},
                {"id", agent_id_},
                {"capabilities", {{"generate", true}}},
                {"tags", {"generator", "rag", "llm"}}
            };
            return card.dump();
        });

        std::thread t([&server](){ server.start(); });
        std::this_thread::sleep_for(std::chrono::seconds(1));

        AgentRegistration reg;
        reg.id = agent_id_;
        reg.name = "Generator";
        reg.address = "http://localhost:" + std::to_string(port_);
        reg.tags = {"generator", "rag"};
        registry_->register_agent(reg);

        std::cout << "[Generator:" << agent_id_ << "] registered on port " << port_ << std::endl;
        t.join();
    }

private:
    std::string handle_jsonrpc(const std::string& body) {
        try {
            auto req = json::parse(body);
            std::string method = req.value("method", "");

            if (method == "generate") {
                auto gr = GenerateRequest::from_json(req["params"]);

                std::string answer;
                // 模拟模式：拼接 Prompt（Phase 2 可替换为 LLM API）
                answer = "[Based on context] Answer to: " + gr.query.substr(0, 100);

                json result;
                result["jsonrpc"] = "2.0";
                result["id"] = req.value("id", json(nullptr));
                result["result"] = GenerateResponse{answer}.to_json();
                return result.dump();
            }

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
};

int main(int argc, char* argv[]) {
    std::string agent_id = "generator-1";
    int port = 9004;
    std::string api_key = "your-api-key";

    if (argc > 1) agent_id = argv[1];
    if (argc > 2) port = std::stoi(argv[2]);
    if (argc > 3) api_key = argv[3];

    std::cout << "[Generator:" << agent_id << "] starting on port " << port << std::endl;

    GeneratorAgent agent(agent_id, port, "http://localhost:8500", api_key);
    agent.start();
    return 0;
}
