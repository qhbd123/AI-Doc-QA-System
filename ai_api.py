# 对接大模型API（支持多模型：豆包、DeepSeek、智谱清言，运行时动态切换，持久化，支持多轮对话，自动降级）
import os
import time
import json
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tiktoken
from typing import List, Dict, Optional

# 从.env 文件中读取环境变量
load_dotenv()

# ---------- 常量 ----------
NO_CONTEXT_MSG = "无相关参考资料"
# 从环境变量读取持久化文件名，默认 "model_provider.json"，用于保存用户通过前端切换的默认模型（例如豆包、DeepSeek、智谱清言），以便服务重启后还能记住上次的选择
MODEL_PROVIDER_FILE = os.getenv("MODEL_PROVIDER_FILE", "model_provider.json")

# ---------- 系统支持的模型提供商名称 ----------
SUPPORTED_PROVIDERS = ["doubao", "deepseek", "zhipu"]

# 默认模型配置，配置错误时自动回退
DEFAULT_PROVIDER = os.getenv("DEFAULT_MODEL_PROVIDER", "doubao").lower()
if DEFAULT_PROVIDER not in SUPPORTED_PROVIDERS:
    print(f"[ai_api] 警告: DEFAULT_MODEL_PROVIDER='{DEFAULT_PROVIDER}' 不支持，已回退为 'doubao'")
    DEFAULT_PROVIDER = "doubao"

# 豆包配置
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
DOUBAO_API_URL = os.getenv("DOUBAO_API_URL")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL")
DOUBAO_MAX_TOKENS = int(os.getenv("DOUBAO_MAX_TOKENS", "8192"))  # 该模型支持的最大token数，用于上下文截断

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192"))

# 智谱清言配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_API_URL = os.getenv("ZHIPU_API_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")
ZHIPU_MAX_TOKENS = int(os.getenv("ZHIPU_MAX_TOKENS", "8192"))

# 超时配置（连接超时, 读取超时）
REQUEST_TIMEOUT = (float(os.getenv("HTTP_CONNECT_TIMEOUT", "5")),
                   float(os.getenv("HTTP_READ_TIMEOUT", "30")))

PROVIDER_CONFIGS = {
    "doubao": {"api_key": DOUBAO_API_KEY, "api_url": DOUBAO_API_URL, "model": DOUBAO_MODEL,
               "max_tokens": DOUBAO_MAX_TOKENS},
    "deepseek": {"api_key": DEEPSEEK_API_KEY, "api_url": DEEPSEEK_API_URL, "model": DEEPSEEK_MODEL,
                 "max_tokens": DEEPSEEK_MAX_TOKENS},
    "zhipu": {"api_key": ZHIPU_API_KEY, "api_url": ZHIPU_API_URL, "model": ZHIPU_MODEL, "max_tokens": ZHIPU_MAX_TOKENS},
}


# ---------- 配置检查与持久化 ----------
# 检查指定模型提供商的配置是否完整
def _is_provider_configured(provider: str) -> bool:
    cfg = PROVIDER_CONFIGS.get(provider)
    if not cfg:
        return False
    return all([cfg["api_key"], cfg["api_url"], cfg["model"]])  # 使用 all() 函数检查 api_key、api_url、model 三个值是否都存在且不为空


# 将当前用户选择的默认模型提供商保存到本地 JSON 文件中，实现模型切换的持久化
def _persist_provider(provider: str):
    try:
        with open(MODEL_PROVIDER_FILE, "w", encoding="utf-8") as f:
            json.dump({"provider": provider}, f)  # 将包含提供商名称的字典 {"provider": provider} 序列化为 JSON 格式并写入文件
    except Exception as e:
        print(f"[ai_api] 保存模型提供商到文件失败: {e}")


# 从本地 JSON 文件中读取上次保存的默认模型提供商名称
def _load_persisted_provider() -> Optional[str]:
    # 检查持久化文件（如 model_provider.json）是否存在
    if not os.path.exists(MODEL_PROVIDER_FILE):
        return None
    try:
        with open(MODEL_PROVIDER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)  # 读取 JSON 内容并解析为 Python 字典
            provider = data.get("provider", "").lower()
            # 检查读取到的模型名称是否在系统支持的提供商列表中，并且该提供商的配置完整
            if provider in SUPPORTED_PROVIDERS and _is_provider_configured(provider):
                return provider
    except Exception as e:
        print(f"[ai_api] 加载模型提供商文件失败: {e}")
    return None


# 加载持久化提供商
# 调用 _load_persisted_provider() 读取 model_provider.json 文件中保存的上次用户切换的默认模型
persisted = _load_persisted_provider()
if persisted and persisted != DEFAULT_PROVIDER:  # 从文件中获取到的默认模型是否存在且是否不是.env文件中的默认模型
    DEFAULT_PROVIDER = persisted  # 将文件中默认模型加载为默认提供商
    print(f"[ai_api] 从持久化文件(model_provider.json)中加载默认提供商: {DEFAULT_PROVIDER}")

if not _is_provider_configured(DEFAULT_PROVIDER):  # 检查最终默认模型的配置完整性
    print(f"[ai_api] 警告：默认模型提供商 '{DEFAULT_PROVIDER}' 配置不完整，API 调用将失败")
else:
    print(f"[ai_api] 初始化完成，默认使用模型提供商: {DEFAULT_PROVIDER}")


# ---------- 重试会话 ----------
def _create_session():
    session = requests.Session()  # 创建一个 requests.Session 对象，该对象会维持会话，在多次请求之间复用连接，提高性能
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])  # 创建一个 urllib3.Retry 对象，配置重试策略，最多重试3次；重试延迟时间为1，即1,2,4；指定重试的HTTP状态码
    session.mount("https://", HTTPAdapter(max_retries=retry))  # 为 requests.Session 对象指定一个自定义的 HTTPAdapter，并将该适配器应用到所有以 https:// 开头的请求上
    return session

# 创建一个全局的 requests.Session 对象，并为其配置了自动重试策略
session = _create_session()


# ---------- Token 计数辅助 ----------
# 使用 OpenAI 的 tiktoken 库精确计算一段文本在指定大模型下会占用的 Token 数量
# 控制 API 调用的 Token 预算，避免超出模型上下文窗口限制，并为成本估算提供数据
def _estimate_tokens(text: str, model_name: str) -> int:
    try:
        # 尝试根据 model_name 获取对应的编码器（Encoder）
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # 回退到 cl100k_base 编码器
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# 将文本截断到大致符合 Token 预算的长度，并在末尾添加截断提示，防止 API 调用因输入过长而失败
def _truncate_context(context: str, max_tokens: int, model_name: str) -> str:
    # 调用 _estimate_tokens 计算原文本的 token 数
    tokens = _estimate_tokens(context, model_name)
    if tokens <= max_tokens:
        return context
    # 计算目标 Token 数与原 Token 数的比例
    ratio = max_tokens / tokens
    # 根据字符数粗略估计截断后的长度：按比例预估的字符数，再乘以 0.9 是留出安全余量
    new_len = int(len(context) * ratio * 0.9)
    # 按字符位置截断字符串
    truncated = context[:new_len]
    print(f"[ai_api] 上下文超限（{tokens} tokens），已截断至 {_estimate_tokens(truncated, model_name)} tokens")
    return truncated + "\n...(内容过长已截断)"


# ---------- 通用 Prompt 构建（单轮兼容）----------
#　根据是否提供了有效的参考资料（content），动态构建用于发送给大模型的提示词（Prompt）
def _build_prompt(question: str, content: str) -> str:
    if content == NO_CONTEXT_MSG or not content.strip():  # 判断是否有参考资料(包括空白字符和字符串)
        return (
            "请直接回答用户的问题。如果你不知道答案，请如实说“我不确定”。\n"
            f"用户问题：{question}"
        )
    else:
        return (
            "只根据下面的资料回答问题。如果资料中没有相关信息，请说“资料中未提及”。\n"
            f"资料：\n{content}\n\n"
            f"问题：{question}"
        )


# ---------- 核心调用函数（失败时抛出异常）----------
# 与具体大模型 API 进行交互的核心执行器
def _call_messages(provider: str, messages: List[Dict]) -> str:
    # 从全局配置字典 PROVIDER_CONFIGS 中获取该提供商的 API 地址、密钥和模型名称
    cfg = PROVIDER_CONFIGS[provider]
    api_url = cfg["api_url"]
    api_key = cfg["api_key"]
    model_name = cfg["model"]
    # 构造请求体，temperature=0.1 使输出更确定、减少随机性
    data = {"model": model_name, "messages": messages, "temperature": 0.1}
    # 构造请求头，使用 Bearer Token 认证：基于 HTTP 请求头的简单身份认证方案
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    start = time.perf_counter()
    try:
        # 使用全局 session（配置了重试机制的 requests.Session）发起 POST 请求，并设置连接超时和读取超时timeout=REQUEST_TIMEOUT
        resp = session.post(api_url, json=data, headers=headers, timeout=REQUEST_TIMEOUT)
        elapsed = time.perf_counter() - start  # 总耗时
        resp.raise_for_status()  # 检查 resp 的 HTTP 状态码，如果状态码是 400 或更高（如 401 未授权、404 未找到、500 服务器错误），则会抛出 requests.exceptions.HTTPError 异常
        answer = resp.json()["choices"][0]["message"]["content"]  # 解析 JSON 响应，提取模型生成的文本即答案
        print(f"[ai_api] {provider} 调用成功 | 耗时 {elapsed:.2f}s | 答案长度 {len(answer)}")
        return answer
    except requests.exceptions.Timeout as e:
        print(f"[ai_api] {provider} 调用超时")
        raise Exception(f"{provider} 调用超时") from e
    except requests.exceptions.ConnectionError as e:
        print(f"[ai_api] {provider} 网络连接失败")
        raise Exception(f"{provider} 网络连接失败") from e
    except requests.exceptions.RequestException as e:
        print(f"[ai_api] {provider} 请求异常: {e}")
        raise Exception(f"{provider} 请求异常: {e}") from e
    except Exception as e:
        print(f"[ai_api] {provider} 未知错误: {e}")
        raise

# 针对不同模型的快捷调用包装器
def _call_messages_doubao(messages): return _call_messages("doubao", messages)


def _call_messages_deepseek(messages): return _call_messages("deepseek", messages)


def _call_messages_zhipu(messages): return _call_messages("zhipu", messages)


# ---------- 单轮调用（保持兼容），针对特定模型的便捷包装----------
def _call_model(provider: str, question: str, content: str) -> str:
    # 将单个问题（question）和参考资料（content）构建成一个 prompt，然后封装成单条 user 消息，最后调用 _call_messages 发送给指定模型并返回答案
    prompt = _build_prompt(question, content)
    messages = [{"role": "user", "content": prompt}]
    return _call_messages(provider, messages)


def _call_doubao(question: str, content: str) -> str:
    return _call_model("doubao", question, content)


def _call_deepseek(question: str, content: str) -> str:
    return _call_model("deepseek", question, content)


def _call_zhipu(question: str, content: str) -> str:
    return _call_model("zhipu", question, content)


# ---------- 自动降级辅助 ----------
# 生成模型自动降级时的尝试顺序列表,动态生成模型降级尝试顺序
def _get_fallback_providers(primary: str) -> List[str]:
    """返回降级尝试顺序（主提供商在前，其余已配置的提供商按固定顺序在后）"""
    all_configured = [p for p in SUPPORTED_PROVIDERS if _is_provider_configured(p)]   # 对每个提供商调用 _is_provider_configured(p) 检查其配置是否完整
    if not all_configured:  # 如果没有一个模型配置完整
        return []
    # 将主提供商移到第一位，其余保持原顺序（或按你希望的优先级）
    ordered = [primary] + [p for p in all_configured if p != primary]
    return list(dict.fromkeys(ordered))  # 去重


# ---------- 主入口（支持多轮对话历史，自动降级）----------
# 接收用户问题、参考资料（检索结果）以及可选的历史消息。
# 根据当前默认模型或指定的模型，按自动降级顺序尝试调用可用的模型提供商。
# 成功时返回模型生成的答案，失败时返回友好的错误提示。
# 支持多轮对话（通过 history_messages 参数）。
def get_ai_answer(question: str, content: str, provider: str = None, history_messages: List[Dict] = None) -> str:
    """
    调用大模型生成答案，支持多轮对话历史，并自动降级到其他可用模型。
    """
    print(f"[ai_api] 收到问题: {question[:50]}...")
    # 确定主提供商，若未指定则使用全局默认；否则转为小写并检查是否在支持列表中
    if provider is None:
        provider = DEFAULT_PROVIDER
    else:
        provider = provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"不支持的模型提供商: {provider}")

    # 获取降级尝试顺序列表
    fallback_chain = _get_fallback_providers(provider)
    # 没有任何模型配置完整
    if not fallback_chain:
        return "没有可用的模型提供商，请检查环境变量配置。"

    last_error = None
    # 遍历降级链，每次尝试一个模型
    for attempt_provider in fallback_chain:
        print(f"[ai_api] 尝试使用提供商: {attempt_provider} (主请求: {provider})")

        # 再次检查当前尝试的模型配置是否完整（防御性编程）
        if not _is_provider_configured(attempt_provider):
            print(f"[ai_api] 提供商 {attempt_provider} 配置不完整，跳过")
            continue

        # 构建系统消息（包含参考资料）
        if content == NO_CONTEXT_MSG or not content.strip():  # # 判断是否有参考资料(包括空白字符和字符串)
            system_content = "请直接回答用户的问题。如果你不知道答案，请如实说“我不确定”。"
        else:
            system_content = f"只根据下面的资料回答问题。如果资料中没有相关信息，请说“资料中未提及”。\n资料：\n{content}"

        #　构建符合 OpenAI API 格式的 messages 数组：1.系统消息；2.历史对话信息；3.当前用户问题
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": question})

        # 据当前尝试的提供商，调用对应的底层 API 函数
        try:
            if attempt_provider == "doubao":
                answer = _call_messages_doubao(messages)
            elif attempt_provider == "deepseek":
                answer = _call_messages_deepseek(messages)
            elif attempt_provider == "zhipu":
                answer = _call_messages_zhipu(messages)
            else:
                continue

            # 如果调用成功，记录降级日志（如果换了模型）
            if attempt_provider != provider:
                print(f"[ai_api] 自动降级: 主模型 {provider} 失败，已切换至 {attempt_provider}")
            return answer

        except Exception as e:
            last_error = e
            print(f"[ai_api] 提供商 {attempt_provider} 调用失败: {e}")
            continue

    # 所有模型都失败
    print(f"[ai_api] 所有模型均调用失败，最后错误: {last_error}")
    return "大模型服务暂时不可用，请稍后重试或联系管理员。"


# ---------- 运行时动态切换 ----------
# 在运行时动态切换系统使用的默认大模型提供商
def set_default_provider(provider: str) -> bool:
    global DEFAULT_PROVIDER
    provider = provider.lower()
    # 检查 provider 是否在系统支持的提供商列表
    if provider not in SUPPORTED_PROVIDERS:
        print(f"[ai_api] 无法切换默认提供商：不支持的 '{provider}'")
        return False
    # 检查该提供商的配置是否完整
    if not _is_provider_configured(provider):
        print(f"[ai_api] 无法切换默认提供商：'{provider}' 配置不完整")
        return False
    DEFAULT_PROVIDER = provider
    # 将新模型名称写入 model_provider.json 文件
    _persist_provider(provider)
    print(f"[ai_api] 默认提供商已切换为: {provider}")
    return True


# 返回当前系统正在使用的默认模型提供商名称
def get_current_provider() -> str:
    return DEFAULT_PROVIDER


# 返回系统支持的所有模型提供商列表
def get_available_providers() -> list:
    return SUPPORTED_PROVIDERS.copy()

# 获取指定模型提供商的配置状态详情
def get_provider_config_status(provider: str) -> dict:
    # 检查指定模型提供商的配置是否完整
    configured = _is_provider_configured(provider)
    # 获取该提供商对应的配置字典
    cfg = PROVIDER_CONFIGS.get(provider, {})
    return {
        "provider": provider,
        "configured": configured,
        "has_api_key": bool(cfg.get("api_key")),
        "has_api_url": bool(cfg.get("api_url")),
        "has_model": bool(cfg.get("model")),
    }

