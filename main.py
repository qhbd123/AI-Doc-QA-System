from fastapi import FastAPI, UploadFile, Request, Form, File
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional, List, Tuple
import tempfile
import os
import traceback
import time
import json
import uuid
from datetime import datetime
from asyncio import Lock

from starlette.middleware.cors import CORSMiddleware

from document_load import load_document
from ai_api import (
    get_ai_answer,
    set_default_provider,
    get_current_provider,
    get_available_providers,
    get_provider_config_status,
)
from cache import get_cache, set_cache, redis_client, get_cache_stats, clear_qa_cache
from retrieval import search_content

# ---------- 常量 ----------
NO_CONTEXT_MSG = "无相关参考资料"
SESSION_EXPIRE_SECONDS = 1800  # 历史会话过期时间:30分钟
MAX_HISTORY_LEN = 5  # 保留最近5轮对话


# ---------- 日志函数 ----------
# 打印格式化的日志，自动包含时间戳、日志级别、调用该函数的函数名以及消息内容
def log(msg: str, level="INFO"):
    import inspect
    # 获取调用者的栈帧，调用 log 的函数的栈帧
    # .f_back：获取当前栈帧的上一级栈帧
    frame = inspect.currentframe().f_back
    # 提取调用者函数名
    # frame.f_code：返回该栈帧对应的代码对象
    func_name = frame.f_code.co_name if frame else "unknown"
    # datetime.now().strftime('%H:%M:%S')：获取当前时间，格式为 时:分:秒
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] [{func_name}] {msg}")


# ---------- 数据结构 ----------
documents_store = {}
knowledge_base = []
kb_lock = Lock()

app = FastAPI(title="AI智能文档问答系统")
log("系统启动，知识库为空")

# 为 FastAPI 应用添加 CORS（跨域资源共享）中间件，用于解决浏览器跨域请求限制问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 构建统一的 API 响应格式,保证所有接口返回的数据结构一致，便于前端解析和处理
def make_response(code: int, status: str, **kwargs):
    #　JSONResponse 是 FastAPI 提供的响应类，会自动将内容转换为 JSON 格式并设置 Content-Type: application/json
    return JSONResponse(
        status_code=code,
        content={"code": code, "status": status, **kwargs}
    )


# 全局异常处理器，用于捕获所有未被其他专门异常处理器捕获的异常
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # exc.__traceback__：获取异常对象的 traceback 信息
    # traceback.extract_tb(...)：将 traceback 对象转换为可读的栈帧列表
    # [-1]异常实际发生的位置
    tb = traceback.extract_tb(exc.__traceback__)[-1]
    # tb.filename：异常发生的文件完整路径,os.path.basename(...)：提取文件名部分
    file_name = os.path.basename(tb.filename)
    # 异常发生的行号
    line_num = tb.lineno
    # 异常发生的函数名（如果发生在顶层，则为 <module>）
    func_name = tb.name
    log(f"❌ 异常捕获 | {file_name}:{line_num} in {func_name}() | {exc}", level="ERROR")
    return make_response(500, "error", msg=f"服务内部错误: {str(exc)}")


# 根路由（/），用于返回前端页面 index.html
# 指定响应类型为 HTMLResponse（FastAPI 提供的用于返回 HTML 内容的响应类）
# 如果不指定，FastAPI 会默认将字符串作为纯文本返回，但使用 HTMLResponse 会设置正确的 Content-Type: text/html
@app.get("/", response_class=HTMLResponse)
async def get_index():
    # os.path.join(... , "index.html")：拼接出 index.html 的完整路径。假设 main.py 在 /app 目录下，则 index_path = "/app/index.html"
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    # 检查文件是否存在 404
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>index.html 未找到</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # 浏览器会渲染该页面
    return HTMLResponse(content=html_content)


# 健康检查接口（/health），用于监控服务的运行状态
@app.get("/health")
async def health():
    return make_response(200, "success", service_status="healthy",
                         total_chunks=len(knowledge_base),
                         redis_available=redis_client is not None)


# 获取 Redis 缓存的运行统计数据，包括总请求次数、命中次数、未命中次数、命中率、平均读取/写入延迟等
@app.get("/cache_stats")
async def cache_stats():
    stats = get_cache_stats()
    return make_response(200, "success", **stats)


# ---------- 会话历史管理 ----------
# 生成 Redis 中存储会话历史记录的键名（key）
def _get_session_key(session_id: str) -> str:
    return f"session:{session_id}"


# 根据会话 ID 从 Redis 中读取该会话的多轮对话历史记录
def _get_history(session_id: str) -> List[dict]:
    if not redis_client:   # 防御性检查：Redis 连接是否成功建立
        log(f"Redis不可用，无法获取会话历史", "WARN")
        return []
    key = _get_session_key(session_id)
    try:
        history_json = redis_client.get(key)
        if history_json:
            # json.loads 将 JSON 字符串反序列化为 Python 列表
            history = json.loads(history_json)
            log(f"获取会话历史成功，共 {len(history)} 条消息")
            return history
        else:
            log(f"会话 {session_id} 无历史记录")
            return []
    except Exception as e:
        log(f"获取会话历史异常: {e}", "ERROR")
        return []

# 会话历史的有界存储和自动过期
# 将会话的历史消息列表保存到 Redis 中，并自动截断只保留最近的 MAX_HISTORY_LEN 轮对话，同时设置过期时间
def _save_history(session_id: str, history: List[dict]):
    if not redis_client:  # 检查全局 Redis 客户端是否可用
        return
    key = _get_session_key(session_id)  # 调用辅助函数生成 Redis 键名
    trimmed = history[-MAX_HISTORY_LEN * 2:]
    try:
        # json.dumps(trimmed) 将历史消息列表序列化为 JSON 字符串，以便存储
        # setex 是 Redis 命令，设置键值对的同时指定过期时间（秒）
        redis_client.setex(key, SESSION_EXPIRE_SECONDS, json.dumps(trimmed))
        log(f"保存会话历史成功，共 {len(trimmed)} 条消息")
    except Exception as e:
        log(f"保存会话历史异常: {e}", "ERROR")


# 多轮对话功能的“写入接口”
# 将一轮新的对话（用户问题 + AI 回答）追加到指定会话的历史记录中
def _add_to_history(session_id: str, user_msg: str, assistant_msg: str):
    history = _get_history(session_id)  # 调用 _get_history 函数，从 Redis 中读取该会话的现有历史消息列表
    # 分别追加用户消息和助手消息
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # 调用 _save_history 函数，将更新后的完整历史列表保存回 Redis
    _save_history(session_id, history)
    log(f"添加对话到历史，当前共 {len(history)} 条消息")


# 精确删除文档及其所有片段的工具函数
# 根据文档 ID 从系统中完全删除该文档的所有知识片段
async def _delete_document_by_id(doc_id: str):
    # 检查 documents_store 字典中是否存在该 doc_id
    if doc_id not in documents_store:
        return 0
    before = len(knowledge_base)
    # 遍历 knowledge_base 中的所有片段，筛选出 doc_id 等于要删除的文档 ID 的片段，生成一个列表 removed
    removed = [item for item in knowledge_base if item["doc_id"] == doc_id]
    # 遍历 removed 列表，对每个片段调用 knowledge_base.remove(item) 将其从原列表中移除
    for item in removed:
        knowledge_base.remove(item)
    after = len(knowledge_base)
    # 从文档元信息字典中删除该文档的条目
    del documents_store[doc_id]
    return before - after


# ---------- 智能问题提取器，用于从 HTTP 请求中获取用户提出的问题 ----------
# 支持多种请求格式：
# 表单数据（application/x-www-form-urlencoded 或 multipart/form-data）：直接读取 form_question 参数。
# JSON 请求体（application/json）：解析 JSON 并获取 question 字段。
# 纯文本请求体（text/plain）：直接将整个请求体作为问题。
# 嵌套 JSON（请求体可能因某些客户端封装而二次序列化，例如 "{ \"question\": \"...\" }" 作为字符串发送）：自动处理并提取。
async def _extract_question(request: Request, form_question: Optional[str]) -> Optional[str]:
    """从表单或 JSON body 中提取问题"""
    if form_question:
        # 去除首尾空格
        return form_question.strip()
    # 尝试将请求体解析为 JSON（application/json）
    try:
        body = await request.json()
        return body.get("question", "").strip()
    # 获取原始请求体（字节串），解码为 UTF-8 字符串
    except:
        try:
            raw = await request.body()
            if raw:
                decoded = raw.decode("utf-8").strip()
                # 果解码后的字符串以 { 开头并以 } 结尾，说明它可能是一个 JSON 字符串（可能因某些客户端错误地将 JSON 作为纯文本发送）。则尝试再次解析 JSON，并提取 question 字段
                if decoded.startswith("{") and decoded.endswith("}"):
                    data = json.loads(decoded)
                    return data.get("question", "").strip()
                else:
                    return decoded
        except:
            return None


# 将表单或 JSON 中的文档 ID 转换为列表
async def _extract_doc_ids(request: Request, form_doc_ids: Optional[str]) -> Optional[List[str]]:
    """从表单或 JSON body 中提取文档 ID 列表"""
    # 如果 form_doc_ids 存在（非 None 且非空字符串），则按逗号 , 分割成多个子串
    # 对每个子串 did.strip() 去除首尾空格，并过滤掉空字符串
    if form_doc_ids:
        return [did.strip() for did in form_doc_ids.split(",") if did.strip()]
    # 尝试将请求体解析为 JSON
    try:
        body = await request.json()
        # 检查解析后的 body 是否为字典，且包含键 "doc_ids"
        if isinstance(body, dict) and "doc_ids" in body:
            # 获取 doc_ids_list，并检查它是否为列表类型
            doc_ids_list = body.get("doc_ids")
            if isinstance(doc_ids_list, list):
                # 转换为字符串
                return [str(did) for did in doc_ids_list]
    except:
        pass
    return None


# 缓存命中的统一处理函数
def _handle_cache_hit(
    req_session_id: str,
    q: str,
    cached_data: dict,
    session_id: str
):
    """缓存命中时，添加历史并返回响应"""
    _add_to_history(req_session_id, q, cached_data["answer"])
    return make_response(200, "success", question=q,
                         answer=cached_data["answer"],
                         from_cache=True,
                         source=cached_data.get("source", "unknown"),
                         sources=cached_data.get("sources", []),
                         session_id=session_id)


# 多文件上传处理器
# 上传一个或多个文档（支持 PDF、DOCX、TXT、MD、HTML），并将文档内容解析、分块后存入系统的知识库（knowledge_base）
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):  # files: List[UploadFile] = File(...)：接收一个或多个上传的文件，使用 FastAPI 的 File 依赖注入，List[UploadFile] 表示支持多文件上传
    log(f"收到上传请求，文件数: {len(files)}")
    allowed_ext = {"pdf", "txt", "docx", "md", "html"}
    uploaded_docs = []
    total_new_chunks = 0

    for file in files:
        # 提取文件扩展名并转为小写
        ext = file.filename.lower().split(".")[-1]
        if ext not in allowed_ext:
            log(f"跳过不支持的文件: {file.filename}")
            continue

        # 读取文件内容到内存；检查文件大小是否超过 10MB
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            log(f"文件过大跳过: {file.filename}")
            continue

        # 创建一个临时文件，将上传的文件内容写入该临时文件。delete=False 表示关闭文件后不自动删除
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 解析文档并分块，返回文本片段列表
            chunks = load_document(tmp_path, max_chunk_size=500)
            # 获取全局锁 kb_lock，保护对 knowledge_base 和 documents_store 的并发写操作
            async with kb_lock:
                existing_doc_id = None
                # 遍历 documents_store，检查是否存在同名文档（文件名相同）。因为允许覆盖，所以需要先删除旧文档
                for doc_id, info in documents_store.items():
                    if info["name"] == file.filename:
                        existing_doc_id = doc_id
                        break
                if existing_doc_id:
                    log(f"发现同名文档 {file.filename}，先删除旧文档")
                    # # 用 _delete_document_by_id 删除旧文档的所有片段及元数据，并记录删除的片段数量
                    removed = await _delete_document_by_id(existing_doc_id)
                    log(f"已删除旧文档，移除 {removed} 个片段")
                # 生成一个新的 UUID 作为文档 ID，并将文档元信息存入 documents_store
                doc_id = str(uuid.uuid4())
                documents_store[doc_id] = {
                    "name": file.filename,
                    "chunks": chunks,
                    "upload_time": time.time()
                }
                # 将每个文本片段（附带文档 ID）追加到全局知识库列表 knowledge_base 中
                for chunk in chunks:
                    knowledge_base.append({"doc_id": doc_id, "text": chunk})
                total_new_chunks += len(chunks)
                uploaded_docs.append({"doc_id": doc_id, "name": file.filename, "chunks": len(chunks)})
                log(f"✅ 上传成功: {file.filename} | 片段数 {len(chunks)}")
        except Exception as e:
            log(f"❌ 文档解析失败: {file.filename} - {e}", level="ERROR")
        finally:
            os.unlink(tmp_path)

    # 没有文档上传成功
    if total_new_chunks == 0:
        return make_response(400, "error", msg="没有有效文件被上传")

    # 调用 clear_qa_cache()，清除所有问答缓存，因为文档内容发生了变化
    clear_qa_cache()
    log(f"上传完成，总计新增 {total_new_chunks} 个片段，已清除缓存")
    return make_response(200, "success",
                         msg=f"成功上传 {len(uploaded_docs)} 个文件",
                         uploaded=uploaded_docs,
                         total_chunks=total_new_chunks,
                         total_in_kb=len(knowledge_base))


# ---------- 删除单个文档 ----------
# 根据文档 ID 删除指定的文档
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in documents_store:
        return make_response(404, "error", msg="文档不存在")
    # 获取全局锁 kb_lock，保护对 knowledge_base 和 documents_store 的写操作（防止并发删除/上传导致数据不一致）
    async with kb_lock:
        # _delete_document_by_id(doc_id)，该函数会从 knowledge_base 中删除该文档的所有片段，并从 documents_store 中删除文档元数据，并返回删除的片段数量（removed）
        removed = await _delete_document_by_id(doc_id)
    log(f"删除文档 {doc_id}，移除 {removed} 个片段")
    # 由于文档已删除，任何与该文档相关的缓存答案都可能误导用户，因此必须清空整个缓存
    clear_qa_cache()
    return make_response(200, "success", msg="文档已删除", removed_chunks=removed)


# 获取当前系统中所有已上传文档的简要信息列表
@app.get("/documents")
async def list_documents():
    # 遍历全局字典 documents_store 的每一项
    docs = [{"doc_id": doc_id, "name": info["name"], "chunks": len(info["chunks"])}
            for doc_id, info in documents_store.items()]
    return make_response(200, "success", documents=docs)


# ---------- 智能问答（重构版）----------
@app.post("/chat")
async def chat(
        request: Request,
        question: Optional[str] = Form(None),
        doc_ids: Optional[str] = Form(None),
        session_id: Optional[str] = Form(None)
):
    start = time.perf_counter()
    log("收到问答请求")

    # 1. 提取问题，用于从 HTTP 请求中获取用户提出的问题
    q = await _extract_question(request, question)
    if not q:
        return make_response(400, "error", msg="请输入问题")

    # 2. 获取或生成 session_id
    # 优先使用表单参数 session_id
    req_session_id = session_id
    # 尝试从 JSON 请求体中获取 session_id
    if not req_session_id:
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_session_id = body.get("session_id", "")
        except:
            pass
    # 根据能否获取session_id生成新旧会话，生成新的 UUID
    if not req_session_id:
        req_session_id = str(uuid.uuid4())
        new_session = True
        log(f"生成新会话: {req_session_id}")
    else:
        new_session = False
        log(f"使用已有会话: {req_session_id}")

    # 3. 提取 doc_ids，将表单或 JSON 中的文档 ID 转换为列表
    selected_doc_ids = await _extract_doc_ids(request, doc_ids)

    # 获取当前使用的模型提供商
    current_provider = get_current_provider()
    # 将问题规范化：去除首尾空格并转为小写，用于缓存 key
    normalized_q = q.strip().lower()

    # 4. 从 Redis 获取历史消息
    history = _get_history(req_session_id) if not new_session else []
    log(f"历史消息数量: {len(history)}")

    # 5. 构建缓存 key，根据文档选择生成签名
    # selected_doc_ids == [] → "no_docs"（全不选）
    # selected_doc_ids 为 None → "all"（全选）
    # 否则将文档 ID 排序后用 _ 连接，例如 "doc1_doc2"。
    # 最终缓存 key 示例："doubao:系统用了哪些技术:docs_all"。
    if selected_doc_ids == []:
        doc_sig = "no_docs"
    else:
        doc_sig = "all" if not selected_doc_ids else "_".join(sorted(selected_doc_ids))
    cache_key = f"{current_provider}:{normalized_q}:docs_{doc_sig}"

    # 6. 始终查询缓存（无论是否有历史）
    cached_raw = get_cache(cache_key)
    if cached_raw:
        try:
            cached_data = json.loads(cached_raw)
            log(f"✅ 缓存命中 | 模型: {current_provider} | 文档选择: {doc_sig} | 问题: {q[:30]}...")
            # 调用 _handle_cache_hit 将答案加入历史并返回响应
            return _handle_cache_hit(req_session_id, q, cached_data, req_session_id)
        except:
            pass

    # 7. 未命中缓存，根据是否有文档选择进行处理
    # 全不选：直接使用 NO_CONTEXT_MSG 作为上下文，标记来源为 "model"
    if selected_doc_ids == []:
        # 全不选：无文档依赖
        context = NO_CONTEXT_MSG
        source = "model"
        source_doc_names = []
        log("全不选模式，调用大模型")
    # 有文档选择（包括全选和部分文档）：调用 search_content 进行 BM25 检索，得到上下文 context 和引用文档名列表 source_doc_names。
    # 若检索无结果，context 仍为 NO_CONTEXT_MSG，此时 source 为 "model"；否则 source = "document"
    else:
        # 有文档选择：检索
        log("缓存未命中，开始检索")
        context, source_doc_names = search_content(
            question=q,
            knowledge_base=knowledge_base,
            documents_store=documents_store,
            doc_ids=selected_doc_ids,
            top_k=3
        )
        source = "document" if context != NO_CONTEXT_MSG else "model"

    # 8. 调用大模型（支持历史）
    answer = get_ai_answer(q, context, history_messages=history)

    # 9. 保存到历史
    _add_to_history(req_session_id, q, answer)

    # 10. 写入缓存（无论是否有历史，都存入），将答案及元数据序列化为 JSON 字符串，存入 Redis 缓存
    cache_data = json.dumps({
        "answer": answer,
        "source": source,
        "sources": source_doc_names if selected_doc_ids != [] else []
    }, ensure_ascii=False)
    set_cache(cache_key, cache_data)

    elapsed = time.perf_counter() - start
    log(f"问答完成，耗时 {elapsed:.3f}s，来源: {source}, 模型: {current_provider}, 引用文档: {source_doc_names if selected_doc_ids != [] else []}")
    return make_response(200, "success", question=q, answer=answer,
                         from_cache=False, source=source,
                         sources=source_doc_names if selected_doc_ids != [] else [],
                         session_id=req_session_id)


# ---------- 清空所有文档 ----------
@app.delete("/clear")
async def clear_all():
    global knowledge_base, documents_store
    async with kb_lock:
        knowledge_base = []
        documents_store = {}
    # 清除所有问答缓存
    clear_qa_cache()
    log("知识库已完全清空，缓存已清除")
    return make_response(200, "success", msg="知识库已清空")


# ---------- 清除所有问答缓存（保留文档） ----------
@app.delete("/clear_cache")
async def clear_cache_only():
    count = clear_qa_cache()
    log(f"已清除 {count} 个问答缓存条目", "INFO")
    return make_response(200, "success", msg=f"已清除 {count} 个缓存条目")


# ---------- 统计信息 ----------
@app.get("/stats")
async def stats():
    return make_response(200, "success",
                         total_chunks=len(knowledge_base),
                         redis_available=redis_client is not None)


# ---------- 模型切换 ----------
@app.get("/models")
async def list_models():
    providers = get_available_providers()
    statuses = [get_provider_config_status(p) for p in providers]
    current = get_current_provider()
    return make_response(200, "success", providers=statuses, current=current)


@app.post("/switch_model")
async def switch_model(provider: str):
    success = set_default_provider(provider)
    if success:
        return make_response(200, "success", msg=f"已切换到 {provider}", current=get_current_provider())
    else:
        return make_response(400, "error", msg=f"无法切换到 {provider}，请检查配置")


@app.get("/current_model")
async def current_model():
    provider = get_current_provider()
    status = get_provider_config_status(provider)
    return make_response(200, "success", current=provider, config_status=status)


# ---------- 会话管理 ----------
@app.get("/session/{session_id}")
async def get_session_history(session_id: str):
    history = _get_history(session_id)
    return make_response(200, "success", history=history)


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if redis_client:
        redis_client.delete(_get_session_key(session_id))
    log(f"会话 {session_id} 历史已清除")
    return make_response(200, "success", msg="会话历史已清除")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    log("AI智能文档问答系统 启动中...")
    log(f"知识库初始片段数: {len(knowledge_base)}")
    log(f"Redis状态: {'已连接' if redis_client else '未连接（将跳过缓存）'}")
    log("API文档: http://127.0.0.1:8001/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app="main:app", reload=False, port=8001)