import PyPDF2

# Path to your PDF file
pdf_file_path = '3207-V0005-1101-0001_EN_Installation_Operations_Maintenance_Manual.pdf'
# Path where you want to save the extracted text
text_file_path = '3207-V0005-1101-0001_EN_Installation_Operations_Maintenance_Manual.txt'

# Open the PDF file in binary mode
with open(pdf_file_path, 'rb') as file:
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(file)
    
    # Initialize an empty string to hold all the text
    full_text = ''
    
    # Loop through each page in the PDF
    for page in pdf_reader.pages:
        # Extract text from the current page
        page_text = page.extract_text()
        
        # Add the text from this page to the full text
        full_text += page_text

# Open a new text file in write mode
with open(text_file_path, 'w', encoding='utf-8') as text_file:
    # Write the extracted text to the file
    text_file.write(full_text)

print(f"Extracted text has been saved to '{text_file_path}'")
