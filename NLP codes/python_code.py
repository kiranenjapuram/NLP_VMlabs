import PyPDF2
pdf_file_path=r"C:\NLP codes\hesc104.pdf"
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text

def process_text(text):
    # Add your data processing logic here
    # Examples: removing unnecessary characters, removing line breaks, etc.
    
    processed_text = text.replace('\n', ' ')
    processed_text = processed_text.replace('\r', '')
    
    return processed_text

# Provide the path to your PDF file
pdf_file_path = r"C:\NLP codes\hesc104.pdf"

# Extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_file_path)

# Process the extracted text
processed_text = process_text(extracted_text)

# Print the processed text
print(processed_text)