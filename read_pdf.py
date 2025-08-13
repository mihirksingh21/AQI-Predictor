import PyPDF2

def read_pdf():
    try:
        with open('Urban Air Quality Prediction Using Satellite Image.pdf', 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            # Read first few pages to understand content
            for i in range(min(3, len(pdf_reader.pages))):
                text = pdf_reader.pages[i].extract_text()
                print(f"\n--- PAGE {i+1} ---")
                print(text[:1000])  # First 1000 characters
                print("...")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    read_pdf() 