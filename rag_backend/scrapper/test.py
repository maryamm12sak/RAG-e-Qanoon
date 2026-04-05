import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load your cleaned OCR data
# Note: Adjust the filepath if you run this from a different directory
with open('scrapper/cleaned_ocr_output.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)

# Initialize counters
total_characters = 0
total_chunks = 0

# Set up the exact splitter you selected for your production configuration
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", "۔", "؟", "!", " ", ""]
)

# Calculate metrics
for doc in documents:
    # Adjust 'text' below if your JSON uses a different key for the content
    text = doc.get('text', '') 
    total_characters += len(text)
    
    chunks = splitter.split_text(text)
    total_chunks += len(chunks)

print(f"Total Documents: {len(documents)}")
print(f"Total Characters: {total_characters}")
print(f"Total Chunks: {total_chunks}")