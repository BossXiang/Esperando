# Esperando
  

> **This project was created for the [AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用](https://tbrain.trendmicro.com.tw/Competitions/Details/37), hosted by E.SUN BANK. The repository includes the code that achieved our best result in the competition. Please follow the instructions below to reproduce our predictions. Note: Data extraction has already been completed, and the corpus is provided in this repo. You may skip the data extraction step and proceed directly to preprocessing and retrieval to save time.**


## Table of Contents

- [Folder Hierarchy](#folder-hierarchy)

- [Environment Setup](#environment-setup)

- [Usage](#usage)
  
---

## Environment Setup

The python version in our development environment is `3.10.7` on `Windows11`

To set up the environment for this project, follow either **Automatic Setup** or **Manual Setup**:

### Automatic Setup

1. **Run the `setup.sh` bash file**
```bash
./setup.sh
```

Note: if this bash file fails, please follow the Manual Setup.

### Manual Setup

1.  **Create a virtual environment**
```bash
python -m venv venv
```

2. **Activate the virtual environment**
```bash
venv\Scripts\activate
```

3.  **Install the required dependencies**
```bash
pip install -r requirements.txt
```

### Other steps

1. **[Required] Acquire Third party API Key**
    - [Data Processing & Retrieval] [Voyage AI API](https://dash.voyageai.com)

2. **[Optional] Acquire Third part API Key and Dependency**
    - [Data extraction] Other dependency [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)
    - [Data extraction] Third party API keys [Google API](https://cloud.google.com/use-cases/ocr)

3.  **Create `.env` file in the project folder** (You may reference [Folder Hierarchy](#folder-hierarchy)) 

4.  **Place the API keys inside the `.env` file following the format below**
```
VOYAGE_API_KEY=YOUR_VOYAGE_API_KEY
```


## Usage


### 1. Data Processing
 - **Working Directory**: `/Preprocess` 
 - **Required Environment Variable**: Set your **Voyage AI API key** in `.env` file

#### Process the data and create clean corpora
```bash
python process_data.py
```


### 2. Retrieval
 - **Working Directory**: `/Model` 
 - **Required Environment Variable**: Set your **Voyage AI API key** in `.env` file

#### Retrieve the document with a given query list
```bash
python retrieve.py
```

#### Retrieve the document with a given query list (Custom I/O)
```bash
python retrieve.py --question_path '../dataset/preliminary/questions_preliminary.json' --dict_path '../Preprocess/datasets/esp_dicts' --output_path 'preds/output.json'
```

**The result (prediction) is stored in the `preds/output.json`**


### 3. [Optional] Evaluation (Not available for the test questions due to lack of ground truths)
 - **Working Directory**: `/Model` 
 - **Required Environment Variable**: N/A

#### Retrieve the document with a given query list
```bash
python evaluate.py
```


### [Optional] Data Extraction (You can skip this step)

**This is to show how we obtain the raw corpus extracted from the pdf files. However, this is not the core to the retrieval logic. Therefore, you might skip this step to save your time.**

 - **Working Directory**: `/Preprocess` 
 - **Required Environment Variable**: Set your Google credentials.
#### **Using Google OCR**
```bash 
export GOOGLE_APPLICATION_CREDENTIALS="../google_credentials.json"
python extract_data_google_ocr.py \
    --project_id your-google-cloud-project-id \
    --location your-processor-location \
    --processor_id your-processor-id \
    --input_base_path ../reference \
    --output_format pickle
```
#### **Using pdfplumber with pytesseract**
```bash 
python extract_data_pdfplumber.py
```

---

## Folder Hierarchy
- `datasets`
  - `preliminary`
    - `ground_truths_example.json`
    - `pred_retrieve.json`
    - `questions_example.json`
    - `questions_preliminary.json`
- `Model`
  - `preds`
    - `output.json`
  - `evaluate.py`
  - `retrieve.py`
- `Preprocess`
  - `datasets` **(Main corpora)**
    - `esp_dicts`
    - `raw_data_ocr`
    - `raw_data_pdfplumber`
  - `others` **(Auxiliary files)**
    - `cn_stopwords.txt`
    - `finance_annotation_accountant.txt`
    - `finance_annotation.txt`
    - `README.md`
  - `extract_data_google_ocr.py`
  - `extract_data_pdfplumber.py`
  - `process_data.py`
  - `utils.py`
- `reference` **(Original dataset in pdf form)**
  - `faq`
  - `finance`
  - `insurance`
- `venv` **(Virtual environment)**
- `.env`
- `.gitignore`
- `README.md`
- `requirements.txt`
- `setup.sh`
