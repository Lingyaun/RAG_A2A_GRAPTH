// demo.cpp -- RAG demo (build + query)
// DAG: Init -> Loader -> Chunker -> Embedder -> QuerySetup -> QueryEmbed -> Search -> Generator
// Fine-grained Params: each stage has its own shared_mutex

#include "../../src/GraphCtrl/GraphInclude.h"
#include "RAGCommon.h"
#include "nodes/DocLoaderNode.h"
#include "nodes/ChunkerNode.h"
#include "nodes/EmbedderNode.h"
#include <EmbeddingClient.h>
#include "nodes/QuerySetupNode.h"
#include "nodes/VectorSearchNode.h"
#include "nodes/LLMGeneratorNode.h"

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string question;
    std::vector<std::string> file_paths;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg.rfind("--query=", 0) == 0) question = arg.substr(8);
            else file_paths.push_back(arg);
        }
    }
    if (file_paths.empty()) file_paths.push_back("test_docs/cpp_concurrency.txt");
    if (question.empty()) question = "What are the concurrency improvements in C++17?";

    CGRAPH_ECHO("[RAG] ============================================");
    CGRAPH_ECHO("[RAG] Demo: %zu file(s), query: %s", file_paths.size(), question.c_str());
    CGRAPH_ECHO("[RAG] ============================================");

    // EmbedderNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    GPipelinePtr pipeline = GPipelineFactory::create();

    GElementPtr init, loader, chunker, doc_embed;
    GElementPtr setup, query_embed, search, generator;

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<DocLoaderNode>(&loader, {init}, "DocLoader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedderNode>(&doc_embed, {chunker}, "DocEmbedder");
    pipeline->registerGElement<QuerySetupNode>(&setup, {doc_embed}, "QuerySetup");
    pipeline->registerGElement<EmbedderNode>(&query_embed, {setup}, "QueryEmbedder");
    pipeline->registerGElement<VectorSearchNode>(&search, {query_embed}, "Search");
    pipeline->registerGElement<LLMGeneratorNode>(&generator, {search}, "Generator");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);
    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);
    dynamic_cast<EmbedderNode*>(query_embed)->setQueryMode(true);

    pipeline->process();

    CGRAPH_ECHO("[RAG] Demo completed.");
    GPipelineFactory::destroy(pipeline);
    return 0;
}