# memory-retriever-xinyi-v1

基于 SQLite FTS5、确定性特征和可选本地 ONNX 重排的对话记忆检索服务，提供同步 Add/Search API，按 user_id 隔离数据。

[English](README.md) · [接口契约](docs/API_CONTRACT.md) · [模型重排](docs/PAIR_RERANK.md) · [索引写入](docs/INDEXED_FTS_WRITES.md)

## 当前发布版本

master 合入了已完成官方 Full 的代码：
`888a3b9b793a3c6c97314cf60ef5d22fcba929aa`。
随后的 README 和 LICENSE 提交只更新文档，运行代码保持一致；冻结引用为
`codex/smoke-20261027-rrf-indexed-parallel2`。

| 官方运行记录 | 结果 |
| --- | --- |
| Textual Full task | `teval_e41f7a2ec090ddd1` |
| Run label | `full-20261027-rrf-v2-888a3b9` |
| 完成时间 | 2026-10-30 12:29:55 UTC+8 |
| 状态 / 总分 | Succeeded / **50.62** |
| Evaluation 进度 | 9602 / 9602 |

该成绩对应上述代码和下方锁定的部署配置，不是文档提交的新评测成绩，也不是公榜排名。
本服务输出检索证据，最终回答和评分由评测平台完成。

## 实现流程

1. **写入：**校验 Add 请求，持久化原始消息，更新 FTS5 和确定性特征后返回成功；同一用户范围内的 request_id 支持幂等。
2. **视图：**保留原消息，同时构造滑动窗口和会话分段，聚合视图通过来源消息 ID 回溯原文。
3. **召回：**FTS5 与有扫描上限的确定性特征通道产生候选；精确词、实体、数字/日期、相对新近度和受保护的更新信号参与排名，options 可参与召回。
4. **重排：**评测配置对前100个候选使用本地 MiniLM cross-encoder 评分，再用 RRF 融合模型排名和原排名。
5. **输出：**去重并保留原消息覆盖，最多返回 top_k 条证据，全程限定 user_id。

FTS 写入使用索引映射，避免按文档 ID 全局扫描。SQLite 使用 WAL、读连接池、串行写锁和有界 busy 重试。
池化 ONNX 后端固定拥有两套独立 tokenizer/模型会话，每次 Search 评分借用一套。

## 本地启动

要求 Python 3.11+，以及启用 FTS5 的 SQLite。默认词法/规则档位仅使用 Python 标准库，模型重排需要显式启用。

```bash
git clone https://github.com/eveiscoding/memory-retriever-xinyi-v1.git
cd memory-retriever-xinyi-v1
python3 -m unittest discover -s tests
python3 scripts/smoke_api.py
python3 -m aml_retriever.cli serve --db ./aml.db --host 127.0.0.1 --port 8080
```

另开终端写入和检索合成示例：

```bash
curl -sS http://127.0.0.1:8080/add \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-1","user_id":"demo-user","session_id":"demo-session","messages":[{"role":"user","content":"My preferred meeting time is Tuesday morning."}]}'

curl -sS http://127.0.0.1:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","query":"When do I prefer meetings?","top_k":10}'
```

Add 返回 success，并原样回传 request_id/user_id/session_id。
Search 返回 data 数组，每项至少包含 id/content，还可包含分数、时间和来源信息。
GET /health 无需认证；对外部署通过 AML_AUTH_MODE=bearer 与 AML_API_KEY 配置认证，管理端点默认关闭。
字段和错误语义见[接口契约](docs/API_CONTRACT.md)。

## Full 评测使用的 ONNX/RRF 配置

| 设置 | 值 |
| --- | --- |
| 模型 | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| 模型快照 revision | `1427fd652930e4ba29e8149678df786c240d8825` |
| 模型文件 | `onnx/model_quint8_avx2.onnx`，CPU INT8 AVX2 |
| query/document 配对上下文 | 512 token，截断 document 一侧 |
| 重排候选池 / 融合 | 100 / RRF，weight=0.25、k=60 |
| 推理 | 两套独立会话，每套 batch=1、threads=1 |
| dense 召回 / embedding 重排 | 该评测配置关闭 |
| 精确命中 / Top20 配对保护 | 该评测配置关闭 |
| 平台 Add / Search 并发上限 | 16 / 16 |

先在绝对路径下准备指定快照的 tokenizer 文件及 ONNX 模型。后端仅加载本地文件，并计算模型资产身份，不会在启动时自动下载模型。
依赖版本与[Dockerfile.onnx](Dockerfile.onnx)一致；完整环境变量和启动命令见英文 README 的 Evaluated ONNX/RRF profile 一节。
关键入口为 `AML_PAIR_BACKEND=aml_retriever.onnx_pair_backend:create_pooled_backend`，
启用 pair_rerank、选择 rrf 模式、设置上述参数，并关闭 vector/semantic_rerank 和两种 pair protection。

## 构建与验证

- Dockerfile：标准库服务镜像。
- Dockerfile.onnx：固定 CPU ONNX 依赖，模型需另行挂载，评测档位需显式配置。
- Dockerfile.rrf-parallel：复用历史冻结依赖环境，需要本地已有 flowgrid-rrf-frozen-base:20261022 镜像。

```bash
python3 -m unittest discover -s tests
python3 -m aml_retriever.cli selfcheck
python3 scripts/smoke_api.py
```

这些是合成 fixture 的功能验证，不代表官方效果。可选后端测试使用 fixture/mock，不能据此推断真实模型性能。
仓库不包含服务数据、密钥、官方输入或评测输出。

## 限制与数据管理

SQLite 是单机存储；cross-encoder 增加延迟，活跃推理由两套模型会话限制。
默认词法档位与评测模型档位的检索行为不同。模型可见的候选正文受配对上下文限制，返回的证据正文可能更长。
时间和更新启发式只影响排名，不会把冲突证据合成为唯一事实。

原消息及来源信息持续保留，支持用户删除和过期清理；官方评测数据还需按每个 run 单独审计保留期，见[生命周期说明](docs/DATA_LIFECYCLE.md)。
模型资产遵循各自的上游许可。

## 来源与许可

原参考仓库：https://github.com/dlxeva/flowgrid-aml-retriever

本仓库采用 [MIT License](LICENSE)，保留原版权声明，并补充本仓库修改部分的版权声明。
