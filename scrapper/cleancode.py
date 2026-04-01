import json
import re
import os

def clean_urdu_legal_text(text: str) -> str:
    """Cleans raw OCR text for Urdu legal documents."""
    
    # 1. Remove specific English headers/footers and OCR hallucinations
    # This removes all English letters (since your text is Urdu)
    text = re.sub(r'[A-Za-z]+', '', text) 
    
    # 2. Remove isolated page numbers and symbols on their own lines (e.g., +25, 626, ١٩٦)
    text = re.sub(r'^\s*[\d۰-۹\+\-\.]+\s*$', '', text, flags=re.MULTILINE)
    
    # 3. Clean up broken sentence markers (e.g., hanging "۔")
    text = re.sub(r'^\s*۔\s*', '۔ ', text, flags=re.MULTILINE)
    
    # 4. Remove excess whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse multiple spaces into one
    text = re.sub(r'\n{3,}', '\n\n', text)  # Collapse 3+ newlines into just 2
    
    return text.strip()

if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, "ocr_output.json")
    output_path = os.path.join(BASE_DIR, "cleaned_ocr_output.json")

    # Load raw OCR data
    print(f"Loading raw data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data = []
    
    for doc in raw_data:
        cleaned_text = clean_urdu_legal_text(doc["text"])
        # Only keep documents that still have text after cleaning
        if len(cleaned_text) > 50:
            cleaned_data.append({
                "text": cleaned_text,
                "meta": doc["meta"]
            })

    # Save cleaned data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully cleaned {len(cleaned_data)} documents.")
    print(f"Saved to: {output_path}")