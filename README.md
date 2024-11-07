# Esperando
  

> Project description (To-be-completed)

  

## Table of Contents

- [Environment Setup](#environment-setup)

- [Usage](#usage)
  
---


## Environment Setup

To set up the environment for this project, follow these steps:

1.  **Create a virtual environment**:
```bash
python -m venv venv
```

2. **Activate the virtual environment**:
```bash
venv\Scripts\activate
```

3.  **Install the required dependencies**:
```bash
pip install -r requirements.txt
```

4. **Other dependencies**:
	a. [Data extraction] [Tesseract-OCR](https://github.com/tesseract-ocr/tesseract)

5. **Third party API keys**:
	a. [Data extraction] [Google API](https://cloud.google.com/use-cases/ocr)
	b. [Data Processing & Retrieval] [Voyage AI API](https://dash.voyageai.com)

**Note:**
If you decide to skip the Data Extraction step and use the already processed data, you only need to obtain **Voyage AI API key**, and place inside the file `.env` following the format below:
```
VOYAGE_API_KEY=YOUR_VOYAGE_API_KEY
```

## Usage

### 1. Data Extraction (You may skip this step and use the provided raw data)
 - **Working Directory**: `/data_center` 
 - **Required Environment Variable**: Set your Google credentials.
#### **Using Google OCR**
```bash 
export GOOGLE_APPLICATION_CREDENTIALS="../google_credentials.json"
python extract_data_google_ocr.py \
    --project_id your-google-cloud-project-id \
    --location your-processor-location \
    --processor_id your-processor-id \
    --input_base_path ../reference \
    --output_format txt
```
#### **Using pdfplumber with pytesseract**
```bash 
python extract_data_pdfplumber.py
```

### 2. Data Processing
 - **Working Directory**: `/data_center` 
 - **Required Environment Variable**: Set your **Voyage AI API key** in `.env` file

#### Process the data and create clean corpora
```bash
python process_data.py
```

### 3. Retrieval
 - **Working Directory**: `/model` 
 - **Required Environment Variable**: Set your **Voyage AI API key** in `.env` file

#### Retrieve the document with a given query list
```bash
python retrieve.py
```

### 4. Evaluation
 - **Working Directory**: `/model` 
 - **Required Environment Variable**: N/A

#### Retrieve the document with a given query list
```bash
python evaluate.py
```

