import PyPDF2
import os

def read_paper_content(file_path: str) -> str:
    """读取论文内容"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 如果论文是 PDF 格式
    if file_path.endswith(".pdf"):
        paper_content = ""
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                paper_content += page.extract_text()
        return paper_content.strip()

    # 如果论文是 TXT 格式
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as text_file:
            return text_file.read().strip()

    # 不支持的文件类型
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")