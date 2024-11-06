import os
import argparse
from tqdm import tqdm
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import json
import pickle
from PyPDF2 import PdfReader, PdfWriter

# 定義 Document AI 的頁面限制
PAGE_LIMIT = 15

# 定義已處理檔案的記錄檔路徑
PROCESSED_FILES_LOG = "processed_files.json"

# 定義拆分後的 PDF 檔案儲存目錄
SPLIT_PDF_DIR = "split_pdfs"

def get_documentai_client(project_id: str, location: str, processor_id: str, processor_version_id: str = None):
    """
    初始化並返回 Document AI 客戶端和處理器名稱。
    """
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if processor_version_id:
        # 處理器版本的完整資源名稱
        name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    else:
        # 處理器的完整資源名稱
        name = client.processor_path(project_id, location, processor_id)

    return client, name

def split_pdf(pdf_path, page_limit=PAGE_LIMIT, split_dir=SPLIT_PDF_DIR):
    """
    將 PDF 拆分為不超過 `page_limit` 頁的小部分。
    返回拆分後的 PDF 檔案路徑列表，並將檔案保存到指定的拆分目錄。
    """
    os.makedirs(split_dir, exist_ok=True)  # 確保拆分目錄存在
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    # 如果頁數在限制內，返回原始檔案路徑
    if num_pages <= page_limit:
        return [pdf_path]

    # 否則，拆分為多個部分
    split_files = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]  # 只保留檔名
    for i in range(0, num_pages, page_limit):
        writer = PdfWriter()
        for j in range(i, min(i + page_limit, num_pages)):
            writer.add_page(reader.pages[j])

        split_filename = f"{base_filename}_part{i // page_limit + 1}.pdf"
        split_filepath = os.path.join(split_dir, split_filename)
        with open(split_filepath, "wb") as f_out:
            writer.write(f_out)
        split_files.append(split_filepath)

    return split_files

def load_processed_files(log_path):
    """
    加載已處理檔案的記錄。
    格式：{"finance/1.pdf", "insurance/2.pdf", ...}
    """
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed_file(log_path, category, filename):
    """
    將已處理的檔案記錄到記錄檔中。
    使用格式：category/filename.pdf
    """
    processed = load_processed_files(log_path)
    processed.add(f"{category}/{filename}")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(processed), f, ensure_ascii=False, indent=4)
    print(f"Recorded processed file: {category}/{filename}")  # 日誌記錄

def process_pdf(client, processor_name, pdf_path, category, mime_type="application/pdf"):
    """
    使用 Document AI 處理單個 PDF 文件並返回提取的文本。
    如果頁數超過限制，則自動拆分並處理每個部分。
    
    Google API 呼叫標記：
    --------------------------------
    """
    # 拆分 PDF
    split_files = split_pdf(pdf_path)

    extracted_texts = []

    for split_file in split_files:
        with open(split_file, "rb") as f:
            pdf_content = f.read()

        raw_document = documentai.RawDocument(content=pdf_content, mime_type=mime_type)

        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=raw_document
        )

        try:
            # 呼叫 Google Document AI API 進行文件處理
            result = client.process_document(request=request)  # Google API 呼叫
            document = result.document
            extracted_texts.append(document.text)
            print(f"Processed and extracted text from: {split_file}")
        except Exception as e:
            print(f"Error processing {split_file}: {e}")
            continue  # 繼續處理下一個拆分部分

    # 合併所有拆分部分的文本
    combined_text = "\n".join(extracted_texts) if extracted_texts else None

    # 清理拆分後的檔案
    for split_file in split_files:
        if split_file != pdf_path and os.path.exists(split_file):
            try:
                os.remove(split_file)
                print(f"Deleted split file: {split_file}")
            except Exception as e:
                print(f"Error deleting split file {split_file}: {e}")

    return combined_text

def save_text(output_dir, filename, text, folder_prefix):
    """
    將提取的文本保存為純文字檔案。
    使用資料夾名稱作為前綴，以避免不同資料夾的檔案名稱衝突。
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]
    full_filename = f"{folder_prefix}_{base_filename}"  # 使用資料夾名稱作為前綴

    # 將文本保存為純文字檔案
    output_path = os.path.join(output_dir, f"{full_filename}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved extracted text as text to {output_path}")

def process_directory(client, processor_name, input_dir, output_dir, category, processed_files=None):
    """
    處理輸入目錄中的所有 PDF 文件，並將提取的文本保存到輸出目錄中。
    跳過已經處理過的文件。
    """
    pdf_files = [file for file in os.listdir(input_dir) if file.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {input_dir}.")
        return

    print(f"Processing {len(pdf_files)} PDF files in {input_dir}...")

    for pdf_file in tqdm(pdf_files):
        # 檢查是否已經處理過該檔案
        log_entry = f"{category}/{pdf_file}"
        if processed_files and log_entry in processed_files:
            print(f"Skipping already processed file: {log_entry}")
            continue

        pdf_path = os.path.join(input_dir, pdf_file)
        extracted_text = process_pdf(client, processor_name, pdf_path, category)
        if extracted_text:
            save_text(output_dir, pdf_file, extracted_text, folder_prefix=category)
            if processed_files is not None:
                save_processed_file(PROCESSED_FILES_LOG, category, pdf_file)

def main():
    parser = argparse.ArgumentParser(description="Process PDFs using Google Document AI and extract text.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud Project ID')
    parser.add_argument('--location', type=str, required=True, help='Processor location, e.g., "us" or "eu"')
    parser.add_argument('--processor_id', type=str, required=True, help='Document AI Processor ID')
    parser.add_argument('--processor_version_id', type=str, default=None, help='Processor Version ID (optional)')
    parser.add_argument('--input_base_path', type=str, required=True, help='Base path containing "finance" and "insurance" directories')
    parser.add_argument('--output_format', type=str, choices=['txt', 'json', 'pickle'], default='txt', help='Output file format')

    args = parser.parse_args()

    client, processor_name = get_documentai_client(
        project_id=args.project_id,
        location=args.location,
        processor_id=args.processor_id,
        processor_version_id=args.processor_version_id
    )

    # 加載已處理的文件記錄
    processed_files = load_processed_files(PROCESSED_FILES_LOG)

    #categories = ['finance', 'insurance']
    categories = ['finance_']
    for category in categories:
        input_dir = os.path.join(args.input_base_path, category)
        output_dir = os.path.join(args.input_base_path, f"{category}_text")
        as_json = args.output_format == 'json'
        as_pickle = args.output_format == 'pickle'
        print(f"\nProcessing category: {category}")
        process_directory(
            client, 
            processor_name, 
            input_dir, 
            output_dir, 
            category=category,  # 資料夾名稱作為前綴
            processed_files=processed_files
        )


    print("\nProcessing completed.")

if __name__ == "__main__":
    main()
