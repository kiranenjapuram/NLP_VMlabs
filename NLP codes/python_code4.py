import PyPDF2
import json

def convert_pdf_to_json(pdf_file, json_file):
    # Open the PDF file in read-binary mode
    with open(pdf_file, 'rb') as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)

        # Initialize an empty list to store the extracted data
        data = []

        # Iterate over each page in the PDF document
        for page_num in range(len(reader.pages)):
            # Get the current page object
            page = reader.pages[page_num]

            # Extract the text from the page
            text = page.extract_text()

            # Split the text into individual lines
            lines = text.split('\n')

            # Remove any empty lines
            lines = [line.strip() for line in lines if line.strip()]

            # Append the lines to the data list
            data.extend(lines)

    # Convert the extracted data to JSON format
    json_data = json.dumps(data, indent=4)

    # Save the JSON data to a file
    with open(json_file, 'w') as file:
        file.write(json_data)

# Specify the input PDF file path
pdf_file = r'C:\NLP codes\hesc104.pdf'

# Specify the output JSON file path
json_file = 'output.json'

# Convert PDF to JSON and save the result
convert_pdf_to_json(pdf_file, json_file)