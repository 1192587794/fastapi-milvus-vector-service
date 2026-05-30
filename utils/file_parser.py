from pathlib import Path


class FileParseError(Exception):
    """文件解析失败时抛出的异常，涵盖格式不支持、文件损坏、内容为空等情况。"""


class FileTextExtractor:
    """从上传文件（PDF、DOCX）中提取纯文本。"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def extract(self, filename: str, content: bytes) -> str:
        """根据文件扩展名分发到对应的解析器。"""
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(content)
        elif suffix == ".docx":
            return self._parse_docx(content)
        else:
            raise FileParseError(
                f"不支持的文件类型: {suffix}，允许: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

    def _parse_pdf(self, content: bytes) -> str:
        """使用 PyMuPDF 提取 PDF 文本。"""
        import fitz

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            raise FileParseError(f"无法打开 PDF 文件: {e}") from e

        pages: list[str] = []
        try:
            for page in doc:
                text = page.get_text()
                if text:
                    pages.append(text)
        finally:
            doc.close()

        if not pages:
            raise FileParseError("PDF 中没有可提取的文本（可能是扫描件或纯图片）。")

        return "\n\n".join(pages)

    def _parse_docx(self, content: bytes) -> str:
        """使用 python-docx 提取 DOCX 文本。"""
        import io

        from docx import Document

        try:
            doc = Document(io.BytesIO(content))
        except Exception as e:
            raise FileParseError(f"无法打开 DOCX 文件: {e}") from e

        paragraphs: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        if not paragraphs:
            raise FileParseError("DOCX 文件中没有文本内容。")

        return "\n\n".join(paragraphs)
