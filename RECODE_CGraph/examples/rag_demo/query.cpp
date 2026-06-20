// query.cpp -- simple query entry
// DAG: Init -> Setup -> QueryEmbed -> Search -> Generator

#include "../../src/GraphCtrl/GraphInclude.h"
#include "RAGCommon.h"
#include "nodes/QuerySetupNode.h"
#include "nodes/EmbedderNode.h"
#include <EmbeddingClient.h>
#include "nodes/VectorSearchNode.h"
#include "nodes/LLMGeneratorNode.h"
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::string question;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if (i > 1) question += " ";
            question += argv[i];
        }
    } else {
        question = "What is C++ concurrency?";
        CGRAPH_ECHO("[RAG] no question, using default.");
    }

    CGRAPH_ECHO("[RAG] ======== query ========");
    // EmbedderNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr init, setup, embed, search, generate;

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<QuerySetupNode>(&setup, {init}, "Setup");
    pipeline->registerGElement<EmbedderNode>(&embed, {setup}, "QueryEmbed");
    pipeline->registerGElement<VectorSearchNode>(&search, {embed}, "Search");
    pipeline->registerGElement<LLMGeneratorNode>(&generate, {search}, "Generator");

    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);
    dynamic_cast<EmbedderNode*>(embed)->setQueryMode(true);

    pipeline->process();

    CGRAPH_ECHO("[RAG] query completed.");
    GPipelineFactory::destroy(pipeline);
    return 0;
}