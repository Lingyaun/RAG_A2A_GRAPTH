// ============================================================
// build_index.cpp — RAG 建库入口
// DAG: Loader → Chunker → Embedder（串行链）
// ============================================================

#include "../../src/GraphCtrl/GraphInclude.h"
#include "RAGCommon.h"
#include "nodes/DocLoaderNode.h"
#include "nodes/ChunkerNode.h"
#include "nodes/EmbedderNode.h"

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

    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr loader  = nullptr;
    GElementPtr chunker = nullptr;
    GElementPtr embedder = nullptr;

    // 注册节点
    pipeline->registerGElement<DocLoaderNode>(&loader, {}, "Loader");
    pipeline->registerGElement<ChunkerNode>(&chunker, {loader}, "Chunker");
    pipeline->registerGElement<EmbedderNode>(&embedder, {chunker}, "Embedder");

    // 通过 dynamic_cast 设置 DocLoaderNode 的文件路径
    dynamic_cast<DocLoaderNode*>(loader)->setPaths(file_paths);

    // 执行 DAG: init → run → deinit
    pipeline->process();

    CGRAPH_ECHO("[RAG] build_index completed.");
    GPipelineFactory::destroy(pipeline);
    return 0;
}
