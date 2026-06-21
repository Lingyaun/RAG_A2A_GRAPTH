# RAG Pipeline — 基于 CGraph DAG + A2A引擎的并行检索增强生成系统

基于 [CGraph](https://github.com/ChunelFeng/CGraph) DAG 计算引擎构建的高性能 C++ RAG（Retrieval-Augmented Generation）系统。

> **现状说明**：CGraph 使用了个人仿写学习的 [RECODE_CGraph](https://github.com/Lingyaun/RECODE_CGraph)。目前该项目已计划废弃——真正的性能消耗在 Embedding API 的 IO 延迟和 LLM API 的响应延迟上，C++ 算完也得等响应（除非 Python 算不完或远超响应时间）。C++ 对比 Python 在纯 RAG 场景下没有明显优势，除非有 ONNX 本地 Embedding、本地精排和本地 LLM 推理。但在什么场景下需要这么快的 CPU 响应？除非科技腾飞了 CPU 也要部署模型、也要用 RAG 和多智能体协作。（正好此仿写也只支持 CPU 不支持 GPU。）

> **坦诚说明**：以下纯当架构演示——
> - 无向量数据库，纯内存存储
> - 余弦相似度 O(N·D) 暴力遍历，无近似索引（HNSW/IVF）
> - CrossEncoder 为硬编码权重 `0.7 * old_score + 0.3 * jaccard`，无模型推理
> - 未做检索精度评估（MRR / Recall@5 / NDCG）
> - 作者纯用来学习RAG设计思路
### 什么是 RAG？

RAG（Retrieval-Augmented Generation，检索增强生成）是一种让 LLM 能够**基于外部知识库**回答问题的技术架构。

```
用户提问："2024年公司财报净利润是多少？"
    │
    ▼
┌─────────────────────────────┐
│   Vector Database           │
│  ┌───┐ ┌───┐ ┌───┐         │
│  │C1 │ │C2 │ │C3 │ ...     │  ← 预先存储的文档向量
│  └───┘ └───┘ └───┘         │
└──────────┬──────────────────┘
           │ 检索相关片段
           ▼
┌─────────────────────────────┐
│   LLM + 检索到的上下文        │  ← "根据公司财报显示..."
│   "2024年净利润为12.3亿"     │
└─────────────────────────────┘
```

**核心流程**（标准 RAG）：

```
文档导入 → 文档切分(Chunk) → 向量化(Embed) → 存入向量库
                                                    ↑
用户提问 → 向量化查询 → 向量检索(Top-K) → 重排序(Rerank)
                                                    ↓
                                            拼接Prompt → LLM生成
```

### 为什么要并行加速？

在真实场景中：

| 阶段 | 瓶颈 | 并行策略 |
|------|------|----------|
| **文档切分** | 一个文档一个文档地切，多个文档可并发 | **文档级并行** |
| **向量化** | 每个 Chunk 调一次 Embedding API，IO 密集型 | **Chunk 级并行** |
| **向量检索** | 单路搜索可能有遗漏 | **多路并行搜索**（多索引、多分片） |
| **重排序** | 多个候选段独立计算相关性分数 | **候选段级并行** |
| **LLM 生成** | 单次调用等待 LLM 返回 | 可做**流式+预生成**（如生成多个候选回答再融合） |

> **RECODE_CGraph 正好解决这个："无依赖的节点并行执行，有依赖的节点顺序执行"**

## 整体架构

```
                      用户查询
                         |
           +-------------+-------------+
           |                           |
      Dense 检索 x N               BM25 检索
   （小chunk余弦相似度）          （小chunk稀疏检索）
           |                           |
           +------ 混合融合Fusion ------+
                  （加权合并去重）
                         |
                  层次索引ParentLookup
              （小chunk -> 父段落映射）
                         |
                CrossEncoder x 2
                （query-chunk精排）
                         |
                    LLM 生成
```

## 三级漏斗检索

| 阶段 | 方法 | 候选数 | 延迟 | 精度 |
|------|------|--------|------|------|
| 粗召回 | BM25（稀疏） | 200 | ~1ms | 低 |
| 语义筛选 | Bi-Encoder（稠密） | 50 | ~10ms | 中 |
| 精排 | Cross-Encoder | 5 | ~50ms | 高 |

## 层次索引（Small-to-Big）

文档通过递归语义切分产出两种粒度的 chunk：
- **小chunk**（128-256 字符）：用于索引和检索，语义聚焦、匹配精准
- **大chunk**（父段落，512-1024 字符）：返回给 LLM，上下文完整

建库时通过 `small_to_parent[]` 映射表建立关联，检索时 O(1) 查表即可从小chunk定位到父段落。

## 核心设计：细粒度节点架构

每个节点**只做一件事**（读 / 计算 / 写），最大化 CGraph 的并行调度能力：

```
EmbedComputeNode  （读 DocParam        -> 计算向量）
       |
EmbedMergeNode    （读中间缓冲区         -> 写 EmbedParam）

SearchComputeNode （读 EmbedParam       -> 计算余弦Top-K）
       |
SearchMergeNode   （读中间缓冲区         -> 写 SearchParam）
```

每对 Compute/Merge 节点通过 `shared_ptr<Intermediate>` 无锁传递数据，**不同节点对持有不同的 `shared_mutex`**，BM25 和 Dense 检索可真正并行执行。

### 拆分 Param 体系

不再使用单一 `RAGParam` + 一把锁，而是 9 个独立 Param，各有独立的读写锁：

| Param | 标识 | 用途 |
|-------|------|------|
| `DocParam` | `"doc"` | 文档内容、小/大chunk、small_to_parent映射 |
| `EmbedParam` | `"embed"` | 文档向量 |
| `QueryParam` | `"query"` | 查询文本及子查询 |
| `QueryEmbedParam` | `"qembed"` | 查询向量 |
| `SearchParam` | `"search"` | Dense检索结果 |
| `BM25Param` | `"bm25"` | BM25检索结果 |
| `FusionParam` | `"fusion"` | 融合结果 |
| `ParentParam` | `"parent"` | 父段落查找结果 |
| `AnswerParam` | `"answer"` | 最终回答 |

## 项目结构

```
RECODE_CGraph/
├── src/                         # CGraph DAG 引擎源码
├── examples/rag_demo/
│   ├── nodes/                   # RAG 节点实现
│   │   ├── DocLoaderNode.h      # 文档加载
│   │   ├── ChunkerNode.h        # 递归语义切块 + 双粒度输出
│   │   ├── EmbedderNode.h       # Embedding Compute/Merge 节点对
│   │   ├── EmbeddingClient.h    # DashScope Embedding API 客户端
│   │   ├── VectorSearchNode.h   # Dense 检索 Compute/Merge
│   │   ├── BM25Node.h           # BM25 检索 Compute/Merge
│   │   ├── FusionRerankerNode.h # 混合融合 Compute/Merge
│   │   ├── ParentLookupNode.h   # 层次索引 Compute/Merge
│   │   ├── CrossEncoderNode.h   # CrossEncoder 精排 Compute/Merge
│   │   ├── LLMGeneratorNode.h   # Prompt 构建 + 回答合并
│   │   ├── QuerySetupNode.h     # 查询设置
│   │   └── QueryDecomposerNode.h# 查询分解
│   ├── demo.cpp                 # 简单建库+查询
│   ├── query.cpp                # 纯查询入口
│   ├── build_index.cpp          # 纯建库入口
│   ├── query_parallel.cpp       # 并行 RAG
│   ├── query_hierarchical.cpp   # 完整层次 RAG（~40节点，8层DAG）
│   ├── benchmark.cpp            # 性能基准测试
│   ├── distributed_demo.cpp     # 分布式多 Agent RAG（概念演示）
│   ├── RAGCommon.h              # 9 个 Param 定义 + 工具函数 + InitNode
│   ├── CMakeLists.txt
│   └── third_party/             # 第三方头文件（curl、nlohmann/json）
└── test_docs/                   # 测试文档
```

## 节点清单

### 单操作节点
- **DocLoaderNode** — 读文件 -> 写 `DocParam.documents`
- **ChunkerNode** — 读文档 -> 递归语义切分 -> 写 `chunks_small/large` + `small_to_parent` 映射
- **QuerySetupNode** — 写 `QueryParam.query`
- **QueryDecomposerNode** — 读查询 -> 生成子查询 -> 写 `QueryParam.sub_queries`

### Compute/Merge 节点对（READ/计算 vs WRITE 分离）

| Compute 节点 | 操作 | -> | Merge 节点 | 操作 |
|---|---|---|---|---|
| **EmbedComputeNode** | 读 DocParam -> 计算向量 | -> | **EmbedMergeNode** | 写 EmbedParam |
| **QueryEmbedComputeNode** | 读 QueryParam -> 计算查询向量 | -> | **QueryEmbedMergeNode** | 写 QueryEmbedParam |
| **SearchComputeNode** | 读 EmbedParam+QEmbedParam -> 余弦Top-K | -> | **SearchMergeNode** | 写 SearchParam |
| **BM25ComputeNode** | 读 DocParam+QueryParam -> BM25打分 | -> | **BM25MergeNode** | 写 BM25Param |
| **FusionComputeNode** | 读 SearchParam+BM25Param -> 加权融合 | -> | **FusionMergeNode** | 写 FusionParam |
| **ParentComputeNode** | 读 FusionParam+DocParam -> small-to-parent映射 | -> | **ParentMergeNode** | 写 ParentParam |
| **CEComputeNode** | 读 ParentParam+DocParam+Query -> Jaccard精排 | -> | **CEMergeNode** | 写 ParentParam（覆盖为精排结果） |
| **PromptBuildNode** | 读 ParentParam+DocParam+Query -> 构建Prompt | -> | **AnswerMergeNode** | 写 AnswerParam |

### 基础设施
- **InitNode** — 在 pipeline 初始化时创建全部 9 个 Param
- **EmbeddingClient** — DashScope Text Embedding API 客户端（libcurl + nlohmann/json），支持批量向量化，无 API key 时自动回退 mock

## 编译

### 环境要求
- C++17 编译器（GCC 8+ / Clang 7+ / MSVC 2019+）
- CMake 3.14+
- libcurl（Windows MinGW 需安装 `mingw-w64-x86_64-curl`）

### 编译全部目标

```bash
cd RECODE_CGraph/examples/rag_demo/build
cmake -S .. -B . -G "MinGW Makefiles"   # 或用 Unix Makefiles / Ninja
cmake --build . -j4
```

产出 6 个可执行文件：`demo.exe`、`query.exe`、`build_index.exe`、`query_parallel.exe`、`query_hierarchical.exe`、`benchmark.exe`。

## 使用示例

### 简单演示
```bash
./demo.exe test_docs/cpp_concurrency.txt --query="C++17有哪些并发改进？"
```

### 完整层次 RAG
```bash
./query_hierarchical.exe test_docs/cpp_concurrency_full.txt --query="对比C++11和C++17的并发特性"
```

### 性能基准测试
```bash
./benchmark.exe --delay=50 --jitter=30 --runs=5
```

## DAG 层次结构（query_hierarchical）

```
第1层: Init -> DocLoader -> Chunker -> EmbedCompute -> EmbedMerge          [串行建库]
第2层: QuerySetup -> QueryDecomposer                                       [查询初始化]
第3层: 4x (QEmbedCompute -> QEmbedMerge)                                  [并行向量化]
第4层: 8x (SearchCompute -> SearchMerge) + (BM25Compute -> BM25Merge)     [10路并行检索]
第5层: FusionCompute -> FusionMerge                                        [混合融合]
第6层: ParentCompute -> ParentMerge                                        [层次索引]
第7层: 2x (CECompute -> CEMerge)                                           [并行精排]
第8层: PromptBuild -> AnswerMerge                                          [生成回答]
```

## 核心技术特点

- **细粒度 DAG 节点**：每个节点只做一件事（读/计算/写），依赖关系精确，并行度最大化
- **拆分 Param 体系**：9 个独立 Param 各自持有 `shared_mutex`，不同检索阶段锁竞争最小化
- **Compute/Merge 分离**：计算节点持读锁做运算，合并节点持写锁做搬运——避免持锁期间做 IO/计算
- **混合检索**：BM25（稀疏关键词）+ Dense（余弦相似度）加权融合，alpha 可配置（默认 0.7）
- **层次索引**：小chunk精准检索 + 父段落完整上下文，O(1) 映射表零开销
- **多级漏斗**：粗召回(200) -> 语义筛选(50) -> 精排(5)，兼顾速度与精度
- **Work-Stealing 线程池**：CGraph 动态负载均衡，支持配置延迟/抖动进行基准测试
- **真实 API 接入**：Embedding 接入 DashScope API（1536 维），无 API key 时自动回退确定性 mock
