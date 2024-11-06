import os
import re
import json
import jieba
import pickle
from enum import Enum
from dotenv import load_dotenv
import voyageai

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = set(f.read().splitlines())
    return lines

def load_finance_titles(filepath):
    finance_titles = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            items = line.replace('  ', ' ').split(' ')
            if len(items) != 4: continue
            source, company, report_type, date = items
            finance_titles[source] = { 'company': company, 'report_type': report_type, 'date': date }
    return finance_titles


def loadPk(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def remove_stopwords(tokens, stopword_list):
    # chinese_number_pattern = r'^[一二三四五六七八九十百千万億兆]+$'
    # tokens = [token for token in tokens if not re.match(chinese_number_pattern, token)] # Remove Chinese numbers
    return [token for token in tokens if token not in stopword_list and len(token) > 1]


def is_pure_punctuation_or_numbers(text):
    pattern = r'^[\d\W]+$'
    return bool(re.fullmatch(pattern, text))


def distribution_analysis(content):
    chinese_characters = len(re.findall(r'[\u4e00-\u9fff]', content))
    english_letters = len(re.findall(r'[a-zA-Z]', content))
    punctuation = len(re.findall(r'[^\w\s]', content))  # Non-alphanumeric and non-space 
    numbers = len(re.findall(r'\d', content))
    return chinese_characters, english_letters, punctuation, numbers


def is_exhaustive_table(content):
    chinese_characters, english_letters, punctuation, numbers = distribution_analysis(content)
    return chinese_characters < english_letters + punctuation + numbers


def date_rewriting(text, season_expansion = False, purify_text = False):
    num_to_cn = {'0': '○', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '10': '十', '11': '十一', '12': '十二'}
    # Convert "2029年" to Taiwan year in Chinese numerals, like "一一八年"
    def convert_year_to_cn(year):
        taiwan_year = int(year) % 1911  # Convert to Taiwan calendar year
        if (taiwan_year < 0 or taiwan_year > 163): return ''    # Invalid year
        taiwan_year_str = str(taiwan_year)
        return ''.join(num_to_cn[digit] for digit in taiwan_year_str) + '年'
    # Convert "1-12月" and "1-31日" to Chinese numerals
    def convert_to_cn(prefix, match, suffix):
        number = int(match.group(1))
        if number <= 10:
            return f"{prefix}{num_to_cn[str(number)]}{suffix}"
        elif number < 20:
            return f"{prefix}十{num_to_cn[str(number % 10)]}{suffix}" if number % 10 != 0 else f"十{suffix}"
        else:
            tens = num_to_cn[str(number // 10)] + "十"
            ones = num_to_cn[str(number % 10)] if number % 10 != 0 else ""
            return f"{prefix}{tens}{ones}{suffix}"

    # Convert years
    text = re.sub(r'(\d{1,4})\s*年', lambda m: convert_year_to_cn(m.group(1)), text)
    # Convert seasons
    text = re.sub(r'第\s*([1-4])\s*季', lambda m: f"第{num_to_cn[m.group(1)]}季", text)
    # Convert months (1-12) to Chinese numerals
    if season_expansion: text = re.sub(r'(1[0-2]|[1-9])\s*月', lambda m: convert_to_cn(f"第{num_to_cn[str((int(m.group(1)) - 1) // 3 + 1)]}季", m, '月'), text)
    else: text = re.sub(r'(1[0-2]|[1-9])\s*月', lambda m: convert_to_cn('', m, '月'), text)
    # Convert days (1-31) to Chinese numerals
    text = re.sub(r'(\d{1,2})\s*日', lambda m: convert_to_cn('', m, '日'), text)
    # To lowercase
    text = text.lower()
    text = text.replace('\n', '')
    while '  ' in text: text = text.replace('  ', ' ')
    if purify_text:     # This is especially used for the finance corpus
        text = re.sub(r'\d+', '', text)     # Remove all the numbers
        # text = re.sub(r'(?<![a-z])\s+|\s+(?![a-z])', '', text)      # Remove all the spaces that are not between English letters
        text = text.replace(' ', '')     # Remove all the spaces
        text = re.sub(r'[!\"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]{3,}', '', text)  # Remove punctuation that appears more than 3 times in a row
    return text


# Parameters
prefix = 'datasets\\raw_data_ocr'
postfix = '.pkl'
output_folder = 'datasets\\esp_dicts'

# Stopwords
stopwords_file = '../data_center/others/cn_stopwords.txt'
stopword_list = load_text(stopwords_file)

# Finacne title dict
class Status(Enum):
    Full = 'full'
    Half = 'half'
    CompanyOnly = 'company_only'
finance_titles_file = 'others/finance_annotation_title.txt'
finance_title_dict = {}
with open(finance_titles_file, 'r', encoding='utf-8') as f:
    raw_text = f.read()
    lines = raw_text.splitlines()
    lines = [ line.strip() for line in lines ]
    text = '\n'.join(lines)
    docs = text.split('\n\n')
    for doc in docs:
        # Document extraction
        doc = doc.strip().splitlines()
        title = doc[0]
        idx = int(title.split(' ')[0])
        company = title.split(' ')[1]
        status = Status.Full
        if len(doc) == 1:   # Company only
            if len(title.split(' ')) == 3: 
                status = Status.Half
                doc_type = title.split(' ')[2]
                content = f'{company} {doc_type}'
            elif len(title.split(' ')) == 2: 
                status = Status.CompanyOnly
                content = company
            else: raise ValueError('Invalid title format')
        else:
            for i in range(1, len(doc)): doc[i] = date_rewriting(doc[i], True)
            content = ' '.join(doc[1:])
            content = f'{company} {content}'
        # Tokenization
        tokens = list(jieba.cut_for_search(content))
        finance_title_dict[idx] = { 'source': idx, 'text': content, 'tokens': tokens, 'metadata': {}, 'status': status }

# Finance
finance_pdfplumber = loadPk('datasets/raw_data_pdfplumber/finance.pkl')
idx = 0
n_fold = 2
sep_window = 120
sep_interval = sep_window // n_fold
num_start, num_end = 0, 1035   # exclusive of num_end
res_dict, idx_dict = {}, {}
for i in range(num_start, num_end):
    filename = f'{prefix}\\finance\\finance_{i}{postfix}'
    content = loadPk(filename)
    content_pdf = finance_pdfplumber[i]

    # Data rewriting
    # content = date_rewriting(content, True, is_exhaustive_table(content))
    # content_pdf = date_rewriting(content_pdf, True, is_exhaustive_table(content))
    content = date_rewriting(content, True, True)
    content_pdf = date_rewriting(content_pdf, True, True)

    # Metadata incorporation
    metadata = {}

    idx_dict[i] = []
    sep_i = sep_interval
    if i in finance_title_dict: 
        if finance_title_dict[i]['status'] == Status.Full: sep_i = 30
        elif finance_title_dict[i]['status'] == Status.Half: sep_i = 60
        elif finance_title_dict[i]['status'] == Status.CompanyOnly: sep_i = 90
        else: raise ValueError('Invalid status')
    # Truncate the content into multiple segments
    for start_idx in range(0, len(content), sep_i):
        chunk = content[start_idx: min(len(content), start_idx + sep_window)]
        if i in finance_title_dict: chunk = f'[{finance_title_dict[i]["text"]}]\n{chunk}'   # Expand with the title
        tokenized_chunk = list(jieba.cut_for_search(chunk))
        tokenized_chunk = remove_stopwords(tokenized_chunk, stopword_list)
        res_dict[idx] = { 'source': i, 'text': chunk, 'tokens': tokenized_chunk, 'metadata': metadata }
        idx_dict[i].append(idx)
        idx += 1
        if start_idx + sep_window >= len(content):
            break
    # PDF plumber expansion
    for start_idx in range(0, len(content_pdf), sep_interval):
        chunk = content_pdf[start_idx: min(len(content_pdf), start_idx + sep_window)]
        tokenized_chunk = list(jieba.cut_for_search(chunk))
        tokenized_chunk = remove_stopwords(tokenized_chunk, stopword_list)
        res_dict[idx] = { 'source': i, 'text': chunk, 'tokens': tokenized_chunk, 'metadata': metadata }
        idx_dict[i].append(idx)
        idx += 1
        if start_idx + sep_window >= len(content_pdf):
            break

output_file = f'{output_folder}\\finance.pkl'
output_idx_file = f'{output_folder}\\finance_idx.pkl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
pickle.dump(res_dict, open(output_file, 'wb'))
pickle.dump(idx_dict, open(output_idx_file, 'wb'))


# Insurance
idx = 0
n_fold = 3
sep_window = 90
sep_interval = sep_window // n_fold
num_start, num_end =  1, 644  # exclusive of num_end
res_dict, idx_dict = {}, {}
for i in range(num_start, num_end):
    filename = f'{prefix}\\insurance\\insurance_{i}{postfix}'
    content = loadPk(filename)
    content = content.lower()

    idx_dict[i] = []
    # Truncate the content into multiple segments
    for start_idx in range(0, len(content), sep_interval):
        chunk = content[start_idx: min(len(content), start_idx + sep_window)]
        tokenized_chunk = list(jieba.cut_for_search(chunk))
        tokenized_chunk = remove_stopwords(tokenized_chunk, stopword_list)
        res_dict[idx] = { 'source': i, 'text': chunk, 'tokens': tokenized_chunk, 'metadata': {} }
        idx_dict[i].append(idx)
        idx += 1
        if start_idx + sep_window >= len(content):
            break
output_file = f'{output_folder}\\insurance.pkl'
output_idx_file = f'{output_folder}\\insurance_idx.pkl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
pickle.dump(res_dict, open(output_file, 'wb'))
pickle.dump(idx_dict, open(output_idx_file, 'wb'))


# FAQ
idx = 0
res_dict, idx_dict = {}, {}
with open(f'{prefix}\\pid_map_content.json', 'rb') as f_s:
    key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
    key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
    for key, value in key_to_source_dict.items():
        idx_dict[key] = []
        for qa in value:
            q = qa['question']
            a = '\n'.join(qa['answers'])
            text = f'[Question]{q}\n[Answers]\n{a}'
            tokenized_content = list(jieba.cut_for_search(text))
            tokenized_content = remove_stopwords(tokenized_content, stopword_list)
            res_dict[idx] = { 'source': key, 'text': text, 'tokens': tokenized_content, 'metadata': {} }
            idx_dict[key].append(idx)
            idx += 1

# Encode the result into embeddings
load_dotenv('../.env')
voyage_api_key = os.getenv("VOYAGE_API_KEY")
batch_size = 30
keys = list(idx_dict.keys())
vo = voyageai.Client()
total_usage = 0
for i in range(0, len(idx_dict), batch_size):
    texts = []
    ids = []
    for key in keys[i: min(i + batch_size, len(idx_dict))]:
        for j in idx_dict[key]:
            texts.append(res_dict[j]['text'])
            ids.append(j)
    doc_embeddings = vo.embed(texts, model="voyage-multilingual-2", input_type="document")
    total_usage += doc_embeddings.total_tokens
    for i in range(len(doc_embeddings.embeddings)):
        res_dict[ids[i]]['embedding'] = doc_embeddings.embeddings[i]
print("Total usage: ", total_usage)

output_file = f'{output_folder}\\faq.pkl'
output_idx_file = f'{output_folder}\\faq_idx.pkl'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
pickle.dump(res_dict, open(output_file, 'wb'))
pickle.dump(idx_dict, open(output_idx_file, 'wb'))
