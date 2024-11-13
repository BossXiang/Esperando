import os
import json
import argparse
import pickle

from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # 設定tesseract.exe的路徑


def faq_to_str(faq):
    ''' formulate the faq content into string '''
    # return ' '.join(i['question'] + ' ' + ' '.join(i['answers']) for i in faq)
    return ' '.join(i['question'] for i in faq)


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    ''' Given a source_path to a pdf, read the pdf, collate the data into a dictionary and return it '''
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf_with_table_image(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    '''
    Vanilla read_pdf function (provided by E.SUN)
    However, this is not used in this project
    '''
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    return pdf_text  # 返回萃取出的文本


def read_pdf_with_table_image(pdf_loc, page_infos: list = None):
    ''' An improvement of the read_pdf function where it captures the tables and images in the pdf. '''
    # Parameters
    image_threshold = 1000  # Threshold for filtering small images

    pdf = pdfplumber.open(pdf_loc)  # Open the specified PDF file

    # If page range is specified, extract that range; otherwise extract all pages
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''

    for page in pages:  # Iterate through each page
        # Extract text from the page
        text = page.extract_text()
        if text:
            pdf_text += text

        # Extract tables from the page
        extracted_tables = page.extract_tables()
        if extracted_tables:
            for table in extracted_tables:
                # Convert the table to string and append to pdf_text
                table_text = '\n'.join(['\t'.join(str(cell) if cell is not None else '' for cell in row) for row in table])
                pdf_text += f'\n\n{table_text}\n'  # Append formatted table text

        # Extract images and perform OCR
        images = page.images
        for img in images:
            # Get the image object
            x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
            
            # Clamp the bounding box coordinates to be within the page bounds
            x0 = max(0, x0)
            top = max(0, top)
            x1 = min(page.width, x1)
            bottom = min(page.height, bottom)

            if (abs(top - bottom) * abs(x0 - x1) >= image_threshold) and (x0 >= 0 and top >= 0 and x1 <= page.width and bottom <= page.height):  # Check if the bounding box is valid
                try: 
                    image = page.within_bbox((x0, top, x1, bottom)).to_image()
                    
                    # Convert the image to a format suitable for OCR
                    image_data = image.original

                    # Use pytesseract to extract text from the image
                    ocr_text = pytesseract.image_to_string(image_data, lang='chi_tra')  # Perform OCR on the image
                    pdf_text += ocr_text  # Append OCR text to the PDF text
                except ValueError as e:
                    print(f"Skipping image due to bounding box error: {e}")
    pdf.close()  # Close the PDF file

    return pdf_text  # Return extracted text



if __name__ == "__main__":
    ''' Main function, used to extract data from the pdf files and save them into dictionaries '''
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--source_path', default='../reference', type=str, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', default='./datasets/raw_data_pdfplumber', type=str, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    args = parser.parse_args()  # 解析參數

    source_path_finance = os.path.join(args.source_path, 'finance')  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

    source_path_insurance = os.path.join(args.source_path, 'insurance')  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
        corpus_dict_faq = {key: faq_to_str(value) for key, value in key_to_source_dict.items()}

    # Output
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'insurance.pkl'), 'wb') as f:
        pickle.dump(corpus_dict_insurance, f)  
    with open(os.path.join(args.output_path, 'finance.pkl'), 'wb') as f:
        pickle.dump(corpus_dict_finance, f)  
    with open(os.path.join(args.output_path, 'faq.pkl'), 'wb') as f:
        pickle.dump(corpus_dict_faq, f)  
