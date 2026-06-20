// build_index.cpp -- build index entry
// DAG: Init -> Loader -> Chunker -> EmbedCompute -> EmbedMerge

#include "../../src/GraphCtrl/GraphInclude.h"
#include "RAGCommon.h"
#include "nodes/DocLoaderNode.h"
#include "nodes/ChunkerNode.h"
#include "nodes/EmbedderNode.h"
#include <EmbeddingClient.h>

#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::vector<std::string> file_paths;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) file_paths.push_back(argv[i]);
    } else {
        CGRAPH_ECHO("[RAG] usage: build_index <file1> [file2...]");
        return 0;
    }

    CGRAPH_ECHO("[RAG] ======== build_index: %zu files ========", file_paths.size());

    // EmbedComputeNode::setEmbeddingClient(std::make_shared<EmbeddingClient>());

    auto emb_buf = std::make_shared<EmbedIntermediate>();

    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr init, loader, chunker, emb_comp, emb_merge;

    pipeline->registerGElement<InitNode>(&init, {}, "Init");
    pipeline->registerGElement<DocLoaderNode>(&loader, {init}, "Loader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedComputeNode>(&emb_comp, {chunker}, "EmbedCompute");
    pipeline->registerGElement<EmbedMergeNode>(&emb_merge, {emb_comp}, "EmbedMerge");

    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);
    dynamic_cast<EmbedComputeNode*>(emb_comp)->setBuffer(emb_buf);
    dynamic_cast<EmbedMergeNode*>(emb_merge)->setBuffer(emb_buf);

    pipeline->process();

    CGRAPH_ECHO("[RAG] build_index completed.");
    GPipelineFactory::destroy(pipeline);
    return 0;
}
