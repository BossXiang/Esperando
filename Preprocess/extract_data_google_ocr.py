import os
import argparse
from tqdm import tqdm
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
import json
import pickle
from PyPDF2 import PdfReader, PdfWriter

# Define the page limit for Document AI processing
PAGE_LIMIT = 15

# Define the log file path for processed files
PROCESSED_FILES_LOG = "processed_files.json"

# Define the directory to store split PDF files
SPLIT_PDF_DIR = "split_pdfs"


def get_documentai_client(project_id: str, location: str, processor_id: str, processor_version_id: str = None):
    """
    Initialize and return the Document AI client and processor name.
    """
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if processor_version_id:
        # Full resource name for the processor version
        name = client.processor_version_path(project_id, location, processor_id, processor_version_id)
    else:
        # Full resource name for the processor
        name = client.processor_path(project_id, location, processor_id)

    return client, name


def split_pdf(pdf_path, page_limit=PAGE_LIMIT, split_dir=SPLIT_PDF_DIR):
    """
    Split a PDF into smaller parts, each not exceeding `page_limit` pages.
    Returns a list of paths to the split PDF files and saves files in the specified split directory.
    """
    os.makedirs(split_dir, exist_ok=True)  # Ensure the split directory exists
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    # If the number of pages is within the limit, return the original file path
    if num_pages <= page_limit:
        return [pdf_path]

    # Otherwise, split into multiple parts
    split_files = []
    base_filename = os.path.splitext(os.path.basename(pdf_path))[0]  # Keep only the filename
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
    Load the record of processed files.
    Format: {"finance/1.pdf", "insurance/2.pdf", ...}
    """
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed_file(log_path, category, filename):
    """
    Record a processed file into the log file.
    Format used: category/filename.pdf
    """
    processed = load_processed_files(log_path)
    processed.add(f"{category}/{filename}")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(list(processed), f, ensure_ascii=False, indent=4)
    print(f"Recorded processed file: {category}/{filename}")  # Log entry


def process_pdf(client, processor_name, pdf_path, category, mime_type="application/pdf"):
    """
    Use Document AI to process a single PDF file and return extracted text.
    If the page count exceeds the limit, automatically split and process each part.
    application/pdf indicates that the content is a PDF document, which helps the API understand how to interpret the file.
    """
    # Split the PDF
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
            # Call Google Document AI API for document processing
            result = client.process_document(request=request)  # Google API call
            document = result.document
            extracted_texts.append(document.text)
            print(f"Processed and extracted text from: {split_file}")
        except Exception as e:
            print(f"Error processing {split_file}: {e}")
            continue  # Continue to the next split part

    # Combine the text from all split parts
    combined_text = "\n".join(extracted_texts) if extracted_texts else None

    # Clean up split files
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
    Save extracted text as a plain text file.
    Uses the folder name as a prefix to avoid conflicts with files from different folders.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]
    full_filename = f"{folder_prefix}_{base_filename}"  # Use the folder name as prefix

    # Save text as a plain text file
    output_path = os.path.join(output_dir, f"{full_filename}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved extracted text as text to {output_path}")


def process_directory(client, processor_name, input_dir, output_dir, category, processed_files=None):
    """
    Process all PDF files in the input directory and save extracted text in the output directory.
    Skips files that have already been processed.
    """
    pdf_files = [file for file in os.listdir(input_dir) if file.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {input_dir}.")
        return

    print(f"Processing {len(pdf_files)} PDF files in {input_dir}...")

    for pdf_file in tqdm(pdf_files):
        # Check if the file has already been processed
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
    """
    Main execution logic of the script. It parses command-line arguments, initializes the Document AI client,
    and processes the PDF files in each specified category one by one.
    """
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

    # Load processed files log
    processed_files = load_processed_files(PROCESSED_FILES_LOG)

    categories = ['finance2','insurance2']
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
            category=category,  # Use folder name as prefix
            processed_files=processed_files
        )


    print("\nProcessing completed.")

if __name__ == "__main__":
    main()
