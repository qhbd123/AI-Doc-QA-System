# retrieval.py - BM25 检索（支持动态文档过滤）
import jieba
from typing import List, Optional, Tuple, Dict
from rank_bm25 import BM25Okapi


NO_CONTEXT_MSG = "无相关参考资料"

def _tokenize(text: str) -> List[str]:
    """对对输入的中文文本进行分词，并过滤掉单字词和空白字符，只保留长度大于 1 的词语"""
    tokens = jieba.cut(text)  # 用结巴分词库的精确模式对 text 进行分词，返回一个可迭代的生成器，产生词语列表，该函数能够识别中文词汇，是中文 NLP 的常用工具
    return [t.strip() for t in tokens if len(t.strip()) > 1]


# 实现 RAG 系统中的检索环节，根据用户问题（可能限定文档 ID 列表）从知识库中选出最相关的片段，并控制返回总长度，以供大模型生成答案
def search_content(
    question: str,
    knowledge_base: List[dict],
    documents_store: Dict[str, dict],
    doc_ids: Optional[List[str]] = None,
    top_k: int = 3,
    max_total_length: int = 1000,
) -> Tuple[str, List[str]]:
    """
    BM25 检索 + 长度限制
    question: 用户问题
    knowledge_base: 知识库列表，每个元素为 {"doc_id": str, "text": str}
    documents_store: 文档元信息字典
    doc_ids: 要检索的文档ID列表，None=全部文档；
                空列表 [] 表示“全不选”，调用方（main.py）会提前处理并直接使用大模型知识，
                因此本函数不会收到空列表。
    top_k: 最多返回多少个片段
    max_total_length: 返回的总上下文最大字符数
    :return: (context_str, source_doc_names)
    """
    print(f"[retrieval] 开始检索，知识库片段数: {len(knowledge_base)}")
    print(f"[retrieval] 问题: {question[:50]}...")

    # 边界情况
    if not knowledge_base:
        return NO_CONTEXT_MSG, []
    # 问题为空或长度小于 2 个字符
    if not question or len(question.strip()) < 2:
        return "请输入更具体的问题", []

    # 过滤文档片段,如果指定了 doc_ids（前端勾选的文档），则只保留这些文档对应的片段；否则保留所有片段
    if doc_ids:
        filtered = [item for item in knowledge_base if item["doc_id"] in doc_ids]
    else:
        filtered = knowledge_base.copy()

    if not filtered:
        print("[retrieval] 过滤后无文档片段")
        return NO_CONTEXT_MSG, []

    # 准备 BM25 语料（分词后的文档列表）
    corpus = []
    # 对每个文档片段调用 _tokenize 进行中文分词，得到词列表，存入 corpus
    for item in filtered:
        tokenized = _tokenize(item["text"])
        corpus.append(tokenized)

    # 构建 BM25 模型,创建 BM25Okapi 对象
    bm25 = BM25Okapi(corpus)

    # 对问题分词
    question_tokens = _tokenize(question)
    if not question_tokens:
        return NO_CONTEXT_MSG, []

    # et_scores 计算每个片段与问题的 BM25 相关性得分,计算 BM25 分数
    scores = bm25.get_scores(question_tokens)

    # 打印分数分布（调试用）
    if scores.size > 0:
        max_score = float(max(scores))
        avg_score = float(sum(scores) / len(scores))
        print(f"[retrieval] BM25 最高分: {max_score:.4f}, 平均分: {avg_score:.4f}")

    # 按分数降序排序，取前 top_k 个候选（但需要考虑长度截断）
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # 组装结果，限制总长度
    result = []
    source_doc_names_set = set()
    total_length = 0

    # 取前 top_k 个得分 > 0 的片段
    for idx, score in indexed_scores[:top_k]:
        if score <= 0:
            continue
        item = filtered[idx]
        content = item["text"]
        doc_id = item["doc_id"]
        # 从 documents_store 中获取文档名
        doc_name = documents_store.get(doc_id, {}).get("name", "未知文档")

        # 判断是否超出长度限制
        if total_length + len(content) > max_total_length:
            remaining = max_total_length - total_length
            # 截断当前片段,保留剩余容量至少 20 字符才加入
            if remaining >= 20:
                truncated = content[:remaining] + "..."
                result.append(f"【文档：{doc_name}】\n{truncated}")
                source_doc_names_set.add(doc_name)
            break
        result.append(f"【文档：{doc_name}】\n{content}")
        source_doc_names_set.add(doc_name)
        total_length += len(content)

    print(f"[retrieval] 检索完成，找到 {len(indexed_scores)} 个匹配片段，返回 {len(result)} 个")
    if not result:
        return NO_CONTEXT_MSG, []
    return "\n\n".join(result), list(source_doc_names_set)
