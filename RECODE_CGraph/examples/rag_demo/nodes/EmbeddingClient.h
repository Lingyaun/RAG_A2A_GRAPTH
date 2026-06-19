#ifndef RAG_EMBEDDING_CLIENT_H
#define RAG_EMBEDDING_CLIENT_H

#include <string>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

using json = nlohmann::json;

/**
 * @brief DashScope Text Embedding API 客户端（libcurl 实现，复用 A2A 传输范式）
 *
 * API: https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
 * 模型: text-embedding-v2 (1536 维)
 *
 * 使用方式：
 *   EmbeddingClient client("your-api-key");          // 显式传入 key
 *   EmbeddingClient client;                          // 从 DASHSCOPE_API_KEY 环境变量读取
 *   auto vecs = client.embed({"hello", "world"});    // 批量向量化
 */
class EmbeddingClient {
public:
    explicit EmbeddingClient(const std::string& api_key = "",
                             const std::string& model = "text-embedding-v2")
        : model_(model)
        , api_url_("https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding")
        , dimension_(1536)
    {
        if (!api_key.empty()) {
            api_key_ = api_key;
        } else {
            const char* env_key = std::getenv("DASHSCOPE_API_KEY");
            api_key_ = env_key ? std::string(env_key) : "";
        }
        curl_global_init(CURL_GLOBAL_DEFAULT);
    }

    ~EmbeddingClient() { curl_global_cleanup(); }

    bool is_configured() const { return !api_key_.empty(); }
    int  dimension()     const { return dimension_; }

    /// 批量文本向量化
    std::vector<std::vector<float>> embed(const std::vector<std::string>& texts) {
        if (texts.empty()) return {};
        if (!is_configured()) throw std::runtime_error("EmbeddingClient: no API key");

        json body = { {"model", model_}, {"input", {{"texts", texts}}} };
        std::string response = post(body.dump());
        json resp = json::parse(response);

        if (resp.contains("code"))
            throw std::runtime_error("API Error: " + resp.value("message", "unknown"));

        std::vector<std::vector<float>> results(texts.size());
        for (auto& emb : resp["output"]["embeddings"]) {
            int idx = emb["text_index"].get<int>();
            if (idx < 0 || idx >= (int)texts.size()) continue;
            for (auto& v : emb["embedding"])
                results[idx].push_back(v.get<float>());
        }
        return results;
    }

    /// 单条文本向量化
    std::vector<float> embed_single(const std::string& text) {
        return embed({text}).at(0);
    }

private:
    static size_t write_cb(void* p, size_t s, size_t n, void* u) {
        ((std::string*)u)->append((char*)p, s * n);
        return s * n;
    }

    std::string post(const std::string& data) {
        CURL* curl = curl_easy_init();
        if (!curl) throw std::runtime_error("curl_easy_init failed");

        std::string resp;
        struct curl_slist* h = nullptr;
        h = curl_slist_append(h, "Content-Type: application/json");
        std::string auth = "Authorization: Bearer " + api_key_;
        h = curl_slist_append(h, auth.c_str());

        curl_easy_setopt(curl, CURLOPT_URL, api_url_.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, h);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

        CURLcode rc = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_slist_free_all(h);
        curl_easy_cleanup(curl);

        if (rc != CURLE_OK)
            throw std::runtime_error(std::string("CURL: ") + curl_easy_strerror(rc));
        if (http_code != 200)
            throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + resp);
        return resp;
    }

    std::string api_key_;
    std::string model_;
    std::string api_url_;
    int         dimension_;
};

#endif
