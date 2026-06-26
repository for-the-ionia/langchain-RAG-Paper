# 文献问答与相似论文检索系统

这个项目使用论文标题、摘要、作者、期刊、发表时间等信息构建知识库，通过 FastAPI 提供接口服务，并使用 LangChain 编排“检索器 -> 提示词 -> Qwen 大语言模型”的问答流程。

## 功能

- 从 Excel 多工作表导入论文数据
- 使用 Qwen/DashScope Embedding 构建 Chroma 向量知识库
- 基于论文摘要进行文献问答
- 返回答案引用来源
- 根据问题、标题或摘要检索相似论文
- 支持按期刊和年份范围过滤

## 项目结构

```text
.
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── config.py        # 环境变量配置
│   ├── data_loader.py   # Excel -> LangChain Document
│   ├── rag.py           # LangChain + Qwen 问答与检索
│   └── schemas.py       # API 请求/响应模型
├── scripts/
│   └── ingest.py        # 构建/重建向量库
└── storage/chroma/      # Chroma 向量库持久化目录
```

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

复制配置文件：

```bash
copy .env.example .env
```

在 `.env` 中填写：

```env
QWEN_API_KEY=你的DashScope_API_Key
```

## 导入你的 Excel 数据

你给出的文件路径是：

```text
C:\Users\XXXXX\XXXXXX\期刊.xlsx
```

运行入库脚本：

```bash
python -m scripts.ingest --excel "C:\Users\XXXXX\XXXXXX\期刊.xlsx" --recreate
```

Excel 支持以下字段名，会自动识别：

- 标题：`篇名`、`标题`、`论文标题`、`title`
- 摘要：`摘要`、`abstract`
- 作者：`作者`、`authors`
- 期刊：`期刊`、`journal`
- 时间：`时间`、`发表时间`、`日期`、`year`
- 引用：`引用`、`被引`、`citations`

当前文件中的核心列为：`篇名`、`期刊`、`时间`、`引用`、`摘要`。

## 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

浏览器打开：

- Swagger 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health

## 接口示例

### 文献问答

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"智慧图书馆的发展趋势是什么？\",\"k\":5}"
```

### 限定期刊问答

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"高校图书馆知识产权服务有哪些问题？\",\"journal\":\"大学图书馆学报\",\"k\":5}"
```

### 相似论文检索

```bash
curl -X POST "http://127.0.0.1:8000/api/similar" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"元宇宙与图书馆智慧服务\",\"k\":5}"
```

## 常见问题

如果启动后提示向量库为空，请先执行入库脚本。

如果 Qwen 调用失败，请检查：

- `.env` 中 `QWEN_API_KEY` 是否正确
- 当前网络是否能访问 DashScope
- `QWEN_MODEL`、`QWEN_EMBEDDING_MODEL` 是否是你账号可用的模型名

## 当前运行配置

本机已切换为 `LLM_PROVIDER=gemini`、`EMBEDDING_PROVIDER=local`：

- 本地 JSON 向量库：`storage/paper_index.json`
- 本地 Hashing Embedding：导入论文时不会把整份 Excel 摘要发送到外部 API
- Gemini：仅在调用 `/api/ask` 生成答案时使用，会发送用户问题和检索到的少量论文片段

启动服务：

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
