import re
import unicodedata

from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    文本清洗流水线，在 embedding 之前对原始文本做标准化处理。

    处理步骤：
    1. 去除首尾空白
    2. Unicode NFC 归一化
    3. 去除控制字符（保留 \\n \\t）
    4. 去除 HTML 标签
    5. 去除 URL
    6. 邮箱脱敏 → [EMAIL]
    7. 手机号脱敏 → [PHONE]
    8. 多余空白合并为单空格
    9. 最终 strip
    """
    # 1. 去除首尾空白
    text = text.strip()

    # 2. Unicode NFC 归一化
    text = unicodedata.normalize("NFC", text)

    # 3. 去除控制字符（保留 \n=0x0a \t=0x09）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4. 去除 HTML 标签
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # 5. 去除 URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # 6. 邮箱脱敏
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]", text)

    # 7. 手机号脱敏（中国大陆手机号：1[3-9]X-XXXX-XXXX）
    text = re.sub(r"1[3-9]\d[-\s]?\d{4}[-\s]?\d{4}", "[PHONE]", text)

    # 8. 多余空白合并为单空格
    text = re.sub(r"\s+", " ", text)

    # 9. 最终 strip
    return text.strip()
