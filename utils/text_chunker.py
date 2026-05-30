import re


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    按段落/句子边界将文本切分为多个 chunk。

    返回列表，每个元素为：
    - id: "{doc_id}::chunk::{index}" 格式的 chunk ID
    - text: chunk 文本
    - parent_id: 原始 doc_id
    - chunk_index: 从 0 开始的序号

    切分策略：
    1. 先按双换行 \\n\\n（段落边界）切分
    2. 超长段落按句号（。！？.!?）切分
    3. 仍超长则按字符硬切
    4. 合并小片段至 chunk_size，新 chunk 带前一 chunk 末尾 chunk_overlap 字符
    """
    if not text:
        return [_make_chunk("", doc_id, 0)]

    # 第一步：按段落切分
    paragraphs = re.split(r"\n\n+", text)

    # 第二步：对每个段落，如果超长则按句子切分
    pieces: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
        else:
            sentences = _split_sentences(para)
            pieces.extend(sentences)

    # 第三步：对仍超长的句子做硬切
    final_pieces: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            final_pieces.append(piece)
        else:
            for i in range(0, len(piece), chunk_size):
                final_pieces.append(piece[i : i + chunk_size])

    # 第四步：合并小片段，带 overlap
    chunks = _merge_pieces(final_pieces, doc_id, chunk_size, chunk_overlap)

    return chunks


def _split_sentences(text: str) -> list[str]:
    """按中英文句号切分句子。"""
    # 匹配中英文句号、感叹号、问号，保留下标用于重组
    parts = re.split(r"(?<=[。！？.!?])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_pieces(
    pieces: list[str],
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """将小片段合并为不超过 chunk_size 的 chunks，带 overlap。"""
    if not pieces:
        return [_make_chunk("", doc_id, 0)]

    chunks: list[dict] = []
    current_text = ""
    chunk_index = 0

    for piece in pieces:
        # 如果当前 chunk 为空，直接加入
        if not current_text:
            current_text = piece
            continue

        # 如果加入这个 piece 不会超长，则追加
        if len(current_text) + 1 + len(piece) <= chunk_size:
            current_text += " " + piece
        else:
            # 保存当前 chunk
            chunks.append(_make_chunk(current_text, doc_id, chunk_index))
            chunk_index += 1

            # 新 chunk 开头带 overlap
            if chunk_overlap > 0 and len(current_text) > chunk_overlap:
                overlap_text = current_text[-chunk_overlap:]
                # 尝试在句子/词边界处截断
                space_pos = overlap_text.find(" ")
                if space_pos > 0:
                    overlap_text = overlap_text[space_pos + 1 :]
                current_text = overlap_text + " " + piece
            else:
                current_text = piece

            # 如果合并后仍超长，做硬切
            while len(current_text) > chunk_size:
                chunks.append(_make_chunk(current_text[:chunk_size], doc_id, chunk_index))
                chunk_index += 1
                current_text = current_text[chunk_size - chunk_overlap :] if chunk_overlap > 0 else current_text[chunk_size:]

    # 保存最后一段
    if current_text:
        chunks.append(_make_chunk(current_text, doc_id, chunk_index))

    return chunks


def _make_chunk(text: str, doc_id: str, index: int) -> dict:
    return {
        "id": f"{doc_id}::chunk::{index}",
        "text": text,
        "parent_id": doc_id,
        "chunk_index": index,
    }
