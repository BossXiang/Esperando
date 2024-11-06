import os
import re
import json
import argparse
import pickle
from enum import Enum
import jieba  # 用於中文文本分詞
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from dotenv import load_dotenv
import voyageai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
# from ckiptagger import WS, POS, NER

# ws = WS("../data_center/data")
# pos = POS("../data_center/data")
# ner = NER("../data_center/data")

class Category(Enum):
    Finance = 'finance'
    Insurance = 'insurance'
    Faq = 'faq'
    Other = 'other'


def remove_punctuation(text):
    # Define a regex pattern to match English punctuation, Traditional Chinese punctuation, and digits
    pattern = r'[！-／：-＠，-．「」『』（）【】、。—・…“”‘’,.\n]*'
    # Use re.sub to replace matches with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def partial_match(source, text):
    i, n = 0, len(source)
    for c in text:
        while i < n and source[i] != c: i += 1
        if i >= n: return False
        i += 1
    return True


def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = set(f.read().splitlines())
    return lines


def date_rewriting(text, season_expansion = False):
    text = text.replace('  ', ' ')
    num_to_cn = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '10': '十', '11': '十一', '12': '十二'}
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
    if season_expansion: text = re.sub(r'(\d{1,2})\s*月', lambda m: convert_to_cn(f"第{num_to_cn[str((int(m.group(1)) - 1) // 3 + 1)]}季", m, '月'), text)
    else: text = re.sub(r'(\d{1,2})\s*月', lambda m: convert_to_cn('', m, '月'), text)
    # Convert days (1-31) to Chinese numerals
    text = re.sub(r'(\d{1,2})\s*日', lambda m: convert_to_cn('', m, '日'), text)
    # To lowercase
    text = text.lower()
    text = text.replace('\n', '')
    while '  ' in text: text = text.replace('  ', ' ')
    return text


# 根據查詢語句和指定的來源，檢索答案
def retrieval_model(q_dict, corpus_dict, index_dict, category):
    if category == Category.Finance:
        return finance_model(q_dict, corpus_dict, index_dict)
    elif category == Category.Insurance:
        return insurance_model(q_dict, corpus_dict, index_dict)
    elif category == Category.Faq:
        return faq_model(q_dict, corpus_dict, index_dict)

DEBUG_MODE = False
# Load auxiliary files
load_dotenv('../.env')
voyage_api_key = os.getenv("VOYAGE_API_KEY")
stopwords_file = '../data_center/others/cn_stopwords.txt'
stopword_list = load_text(stopwords_file)

finance_accountant_file = '../data_center/others/finance_annotation_accountant.txt'
finance_accountant_dict = {}
with open(finance_accountant_file, 'r', encoding='utf-8') as f:
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
        for i in range(1, len(doc)):
            doc[i] = date_rewriting(doc[i], True)
        content = '\n'.join(doc[1:])
        content = f'{company}\n{content}'
        # Tokenization
        tokens = list(jieba.cut_for_search(content))
        finance_accountant_dict[idx] = { 'source': idx, 'text': content, 'tokens': tokens, 'metadata': {} }

total_usage = 0
def finance_model(q_dict, corpus_dict, index_dict):
    # Query preprocessing
    query = q_dict['query']
    query = date_rewriting(query)
    query_tokens = list(jieba.cut_for_search(query))
    query_tokens = [ token for token in query_tokens if token not in stopword_list and len(token) > 1 ] 
    # query_metadata = q_dict['metadata']

    # Document preprocessing
    source = q_dict['source']
    filtered_corpus = []
    tokenized_corpus = []

    # Data preparation
    for file in source:
        if file in finance_accountant_dict:
            filtered_corpus.append(finance_accountant_dict[file])
            tokens = finance_accountant_dict[file]['tokens']
            tokenized_corpus.append(tokens)
        for i in index_dict[file]:
            filtered_corpus.append(corpus_dict[i])
            tokens = corpus_dict[i]['tokens']
            tokenized_corpus.append(tokens)
    
    # Initial Retrieval
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    ans = bm25.get_top_n(query_tokens, list(filtered_corpus), n=15)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    res = [ a['source'] for a in ans ]

    # Final reranking (Without embedding-based retrieval first)
    global total_usage
    documents = [ a['text'] for a in ans ]
    res_source = {}
    for a in ans: res_source[a['text']] = a['source']
    vo = voyageai.Client()
    reranking = vo.rerank(query, documents, model="rerank-2", top_k=1)
    total_usage += reranking.total_tokens
    result = reranking.results[0]
    res = res_source[result.document]
    return res


def insurance_model(q_dict, corpus_dict, index_dict):
    # Query preprocessing
    query = q_dict['query']
    query_tokens = list(jieba.cut_for_search(query))
    query_tokens = [ token for token in query_tokens if token not in stopword_list and len(token) > 1 ] 
    # query_metadata = q_dict['metadata']

    # Document preprocessing
    source = q_dict['source']
    filtered_corpus = []
    tokenized_corpus = []
    # Data preparation
    for file in source:
        for i in index_dict[file]:
            filtered_corpus.append(corpus_dict[i])
            tokens = corpus_dict[i]['tokens']
            tokenized_corpus.append(tokens)
    
    # Initial Retrieval
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    ans = bm25.get_top_n(query_tokens, list(filtered_corpus), n=7)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    res = [ a['source'] for a in ans ]

    # Final reranking
    global total_usage
    documents = [ a['text'] for a in ans ]
    res_source = {}
    for a in ans: res_source[a['text']] = a['source']
    vo = voyageai.Client()
    reranking = vo.rerank(query, documents, model="rerank-2", top_k=1)
    total_usage += reranking.total_tokens
    result = reranking.results[0]
    res = res_source[result.document]
    return res


def faq_model(q_dict, corpus_dict, index_dict):
    # Query preprocessing
    query = q_dict['query']

    # Document preprocessing
    source = q_dict['source']
    content_list = []
    embedding_list = []
    source_list = []
    for i in source:
        for j in index_dict[i]:
            content_list.append(corpus_dict[j]['text'])
            embedding_list.append(corpus_dict[j]['embedding'])
            source_list.append(i)

    global total_usage
    vo = voyageai.Client()
    # Initial retrieval
    k = 7
    query_embeddings = q_dict['embedding']
    similarities = cosine_similarity([query_embeddings], embedding_list)
    similarities = similarities[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    res = [source_list[idx] for idx in top_k_indices]
    res_text = [content_list[idx] for idx in top_k_indices]
    res_source = {}
    for i in range(len(res)): res_source[res_text[i]] = res[i]

    # Final reranking
    documents = res_text
    reranking = vo.rerank(query, documents, model="rerank-2", top_k=1)
    total_usage += reranking.total_tokens
    result = reranking.results[0]
    res = res_source[result.document]
    return res


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', default='../dataset/preliminary/questions_example.json', type=str, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--dict_path', default='../data_center/datasets/esp_dicts', type=str, help='The path to the dicts (created by preprocess.py)')  # 參考資料的路徑
    parser.add_argument('--output_path', default='preds/output.json', type=str, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑
    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典
    # Load questions
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    # Preprocess questions
    qs_dict = qs_ref['questions']
    # Obtain embeddings for questions
    vo = voyageai.Client()
    batch_size = 100
    print('Encoding queries...')
    for i in range(0, len(qs_dict), batch_size):
        if i + batch_size > len(qs_dict): batch = qs_dict[i:]
        else: batch = qs_dict[i: i + batch_size]
        query_list = [ q['query'] for q in batch ]
        query_embeddings = vo.embed(query_list, model="voyage-multilingual-2", input_type="query")
        if len(batch) != len(query_embeddings.embeddings): print("Mismatch in batch size")
        for j in range(len(query_embeddings.embeddings)):
            qs_dict[i + j]['embedding'] = query_embeddings.embeddings[j]
            total_usage += query_embeddings.total_tokens

    print('Loading corpora...')
    # Load dicts
    with open(os.path.join(args.dict_path, 'insurance.pkl'), 'rb') as f:
        corpus_dict_insurance = pickle.load(f)
    with open(os.path.join(args.dict_path, 'finance.pkl'), 'rb') as f:
        corpus_dict_finance = pickle.load(f)
    with open(os.path.join(args.dict_path, 'faq.pkl'), 'rb') as f:
        corpus_dict_faq = pickle.load(f)

    # Load index dicts
    with open(os.path.join(args.dict_path, 'insurance_idx.pkl'), 'rb') as f:
        corpus_idx_dict_insurance = pickle.load(f)
    with open(os.path.join(args.dict_path, 'finance_idx.pkl'), 'rb') as f:
        corpus_idx_dict_finance = pickle.load(f)
    with open(os.path.join(args.dict_path, 'faq_idx.pkl'), 'rb') as f:
        corpus_idx_dict_faq = pickle.load(f)

    total_usage = 0
    print('Retrieving answers...')
    # Retrieve answers
    for q_dict in tqdm(qs_dict, desc="Retrieval in progress"):
        if q_dict['category'] == 'finance':
            retrieved = retrieval_model(q_dict, corpus_dict_finance, corpus_idx_dict_finance, Category.Finance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = retrieval_model(q_dict, corpus_dict_insurance, corpus_idx_dict_insurance, Category.Insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            retrieved = retrieval_model(q_dict, corpus_dict_faq, corpus_idx_dict_faq, Category.Faq)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    
    print(f"Total usage: {total_usage}")
    # 將答案字典保存為json文件
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符
    print(f'Retrieval successfully completed! The result is saved at {args.output_path}')
