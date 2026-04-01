import os
import glob
import json
import gc
import io
from google.cloud import vision
from pdf2image import convert_from_path, pdfinfo_from_path

# ── Set Credentials ──
# Make sure "service_account.json" is placed right next to this script in the rag_backend folder
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

def run_ocr_on_folder(folder_path: str, poppler_path: str):
    """
    Reads every PDF using Google Cloud Vision OCR (Urdu + English).
    """
    client = vision.ImageAnnotatorClient()

    texts = []
    metadata = []

    # Find all PDFs in the given folder
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{folder_path}'.")

    print(f"Found {len(pdf_files)} PDF file(s). Starting OCR...\n")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            # IMPORTANT: Pass poppler_path here
            info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
            total_pages = info["Pages"]
            print(f"  Processing {filename} — {total_pages} pages...")

            full_text = ""

            for page_num in range(1, total_pages + 1):
                print(f"    OCR page {page_num}/{total_pages}...")

                # IMPORTANT: Pass poppler_path here as well
                page_img = convert_from_path(
                    pdf_path, 
                    dpi=300, 
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=poppler_path
                )[0]

                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                page_img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

                # Call Google Vision
                image = vision.Image(content=img_byte_arr)
                response = client.document_text_detection(image=image)

                if response.error.message:
                    print(f"    Vision API error: {response.error.message}")
                    continue

                page_text = response.full_text_annotation.text
                if page_text:
                    full_text += page_text + "\n\n"

                del page_img, img_byte_arr, response
                gc.collect()

            if len(full_text.strip()) > 50:
                texts.append(full_text)
                metadata.append({"source": filename, "pages": total_pages})
                print(f"  ✓ Done: {filename}  ({total_pages} pages)\n")
            else:
                print(f"  ⚠ No text extracted: {filename}\n")

        except Exception as e:
            print(f"  ✗ Error in {filename}: {e}\n")

    print(f"OCR complete. {len(texts)} document(s) extracted.")
    return texts, metadata


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Get the absolute path of the directory containing this script (rag_backend)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Go ONE LEVEL UP (acts like ../)
    PARENT_DIR = os.path.dirname(BASE_DIR)

    # 1. Define your paths using the PARENT_DIR
    DATA_FOLDER = os.path.join(PARENT_DIR, "scrapper", "data", "raw")
    OUTPUT_PATH = os.path.join(PARENT_DIR, "scrapper", "ocr_output.json")
    
    # 2. Define the exact path to your Poppler bin folder
    POPPLER_BIN_PATH = r"C:\Uni stuff\Uni stuff\NLP\project\RAG-e-Qanoon\rag_backend\poppler-25.12.0\Library\bin"

    print(f"Looking for PDFs in exact path: {DATA_FOLDER}")

    # Ensure the output directory exists so it doesn't crash when saving
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 3. Run the extraction
    texts, metadata = run_ocr_on_folder(DATA_FOLDER, POPPLER_BIN_PATH)

    ocr_output = [
        {"text": t, "meta": m}
        for t, m in zip(texts, metadata)
    ]

    # 4. Save the results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(ocr_output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(ocr_output)} documents → {OUTPUT_PATH}")