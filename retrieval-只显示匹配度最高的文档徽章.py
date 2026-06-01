# retrieval.py
import jieba
from typing import List, Optional, Tuple, Dict

NO_CONTEXT_MSG = "无相关参考资料"

def search_content(
    question: str,
    knowledge_base: List[dict],
    documents_store: Dict[str, dict],
    doc_ids: Optional[List[str]] = None,
    top_k: int = 3,
    max_total_length: int = 1000,
) -> Tuple[str, List[str]]:
    """
    关键词匹配 + 相关性排序 + 长度限制，返回 (context, source_doc_names)
    其中 source_doc_names 只包含匹配分数最高的文档名（主要来源）
    """
    print(f"[retrieval] 开始检索，知识库片段数: {len(knowledge_base)}")
    print(f"[retrieval] 问题: {question[:50]}...")

    # 将空列表视为全部文档
    if doc_ids == []:
        doc_ids = None
    if doc_ids:
        print(f"[retrieval] 限定文档: {doc_ids}")

    if not knowledge_base:
        return NO_CONTEXT_MSG, []
    if not question or len(question.strip()) < 2:
        return "请输入更具体的问题", []

    question_words = [w.strip() for w in jieba.cut(question) if len(w.strip()) > 1]
    if not question_words:
        return NO_CONTEXT_MSG, []

    if doc_ids:
        filtered = [item for item in knowledge_base if item["doc_id"] in doc_ids]
    else:
        filtered = knowledge_base.copy()
    if not filtered:
        print("[retrieval] 过滤后无文档片段")
        return NO_CONTEXT_MSG, []

    scored_chunks = []
    for item in filtered:
        content = item["text"]
        score = sum(1 for w in question_words if w in content)
        if score > 0:
            scored_chunks.append((score, item["doc_id"], content))
####################################################################################################################
    if not scored_chunks:
        print("[retrieval] 未找到匹配片段")
        return NO_CONTEXT_MSG, []

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    top = scored_chunks[:top_k]

    # 构建上下文（仍然使用所有 top_k 片段）
    result = []
    total_length = 0
    for score, doc_id, content in top:
        doc_name = documents_store.get(doc_id, {}).get("name", "未知文档")
        if total_length + len(content) > max_total_length:
            remaining = max_total_length - total_length
            if remaining >= 20:
                truncated = content[:remaining] + "..."
                result.append(f"【文档：{doc_name}】\n{truncated}")
            break
        result.append(f"【文档：{doc_name}】\n{content}")
        total_length += len(content)

    # 主要来源：最高分片段的文档名
    primary_doc_id = top[0][1]
    primary_doc_name = documents_store.get(primary_doc_id, {}).get("name", "未知文档")
    source_doc_names = [primary_doc_name]

    print(f"[retrieval] 检索完成，找到 {len(scored_chunks)} 个匹配片段，返回 {len(result)} 个，主要来源: {primary_doc_name}")
    return "\n\n".join(result) if result else NO_CONTEXT_MSG, source_doc_names