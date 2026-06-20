// query.cpp -- simple query entry
// DAG: Init -> Setup -> QEmbedCompute -> QEmbedMerge
//      -> SearchCompute -> SearchMerge -> PromptBuild -> AnswerMerge

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
    // EmbedComputeNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    auto qemb_buf   = std::make_shared<QEmbedIntermediate>();
    auto search_buf = std::make_shared<SearchIntermediate>();
    auto answer_buf = std::make_shared<AnswerIntermediate>();

    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr init, setup, qemb_comp, qemb_merge, search_comp, search_merge;
    GElementPtr prompt_build, answer_merge;

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<QuerySetupNode>(&setup, {init}, "Setup");
    pipeline->registerGElement<QueryEmbedComputeNode>(&qemb_comp, {setup}, "QEmbedCompute");
    pipeline->registerGElement<QueryEmbedMergeNode>(&qemb_merge, {qemb_comp}, "QEmbedMerge");
    pipeline->registerGElement<SearchComputeNode>(&search_comp, {qemb_merge}, "SearchCompute");
    pipeline->registerGElement<SearchMergeNode>(&search_merge, {search_comp}, "SearchMerge");
    pipeline->registerGElement<PromptBuildNode>(&prompt_build, {search_merge}, "PromptBuild");
    pipeline->registerGElement<AnswerMergeNode>(&answer_merge, {prompt_build}, "AnswerMerge");

    dynamic_cast<QuerySetupNode*>(setup)->setQuery(question);
    dynamic_cast<QueryEmbedComputeNode*>(qemb_comp)->setBuffer(qemb_buf);
    dynamic_cast<QueryEmbedMergeNode*>(qemb_merge)->setBuffer(qemb_buf);
    dynamic_cast<SearchComputeNode*>(search_comp)->setBuffer(search_buf);
    dynamic_cast<SearchMergeNode*>(search_merge)->setBuffer(search_buf);
    dynamic_cast<PromptBuildNode*>(prompt_build)->setBuffer(answer_buf);
    dynamic_cast<AnswerMergeNode*>(answer_merge)->setBuffer(answer_buf);

    pipeline->process();

    CGRAPH_ECHO("[RAG] query completed.");
    GPipelineFactory::destroy(pipeline);
    return 0;
}
