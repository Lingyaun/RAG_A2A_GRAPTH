// demo.cpp -- RAG demo (build + query)
// DAG: Init -> Loader -> Chunker -> EmbedCompute -> EmbedMerge
//      -> QuerySetup -> QEmbedCompute -> QEmbedMerge
//      -> SearchCompute -> SearchMerge -> PromptBuild -> AnswerMerge
// Fine-grained nodes: each node does ONE thing (READ/COMPUTE/WRITE separation)

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

    // EmbedComputeNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    GPipelinePtr pipeline = GPipelineFactory::create();

    // Intermediate buffers
    auto emb_buf    = std::make_shared<EmbedIntermediate>();
    auto qemb_buf   = std::make_shared<QEmbedIntermediate>();
    auto search_buf = std::make_shared<SearchIntermediate>();
    auto answer_buf = std::make_shared<AnswerIntermediate>();

    GElementPtr init, loader, chunker, emb_comp, emb_merge;
    GElementPtr setup, qemb_comp, qemb_merge, search_comp, search_merge;
    GElementPtr prompt_build, answer_merge;

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<DocLoaderNode>(&loader, {init}, "DocLoader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedComputeNode>(&emb_comp, {chunker}, "EmbedCompute");
    pipeline->registerGElement<EmbedMergeNode>(&emb_merge, {emb_comp}, "EmbedMerge");
    pipeline->registerGElement<QuerySetupNode>(&setup, {emb_merge}, "QuerySetup");
    pipeline->registerGElement<QueryEmbedComputeNode>(&qemb_comp, {setup}, "QEmbedCompute");
    pipeline->registerGElement<QueryEmbedMergeNode>(&qemb_merge, {qemb_comp}, "QEmbedMerge");
    pipeline->registerGElement<SearchComputeNode>(&search_comp, {qemb_merge}, "SearchCompute");
    pipeline->registerGElement<SearchMergeNode>(&search_merge, {search_comp}, "SearchMerge");
    pipeline->registerGElement<PromptBuildNode>(&prompt_build, {search_merge}, "PromptBuild");
    pipeline->registerGElement<AnswerMergeNode>(&answer_merge, {prompt_build}, "AnswerMerge");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);
    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);
    dynamic_cast<EmbedComputeNode*>(emb_comp)->setBuffer(emb_buf);
    dynamic_cast<EmbedMergeNode*>(emb_merge)->setBuffer(emb_buf);
    dynamic_cast<QueryEmbedComputeNode*>(qemb_comp)->setBuffer(qemb_buf);
    dynamic_cast<QueryEmbedMergeNode*>(qemb_merge)->setBuffer(qemb_buf);
    dynamic_cast<SearchComputeNode*>(search_comp)->setBuffer(search_buf);
    dynamic_cast<SearchMergeNode*>(search_merge)->setBuffer(search_buf);
    dynamic_cast<PromptBuildNode*>(prompt_build)->setBuffer(answer_buf);
    dynamic_cast<AnswerMergeNode*>(answer_merge)->setBuffer(answer_buf);

    pipeline->process();

    CGRAPH_ECHO("[RAG] Demo completed.");
    GPipelineFactory::destroy(pipeline);
    return 0;
}
