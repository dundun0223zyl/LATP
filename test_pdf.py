import PyPDF2
import sys
import os

pdf_path = sys.argv[1]
print(f"Testing PDF extraction on: {pdf_path}")
print(f"File exists: {os.path.exists(pdf_path)}")
print(f"File size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'} bytes")

try:
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        print(f"Total pages: {len(reader.pages)}")
        
        # Try to find budget information
        budget_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and any(term in text.lower() for term in ["budget", "£", "million", "allocation", "revenue"]):
                budget_pages.append(i+1)
                print(f"\n--- PAGE {i+1} (CONTAINS BUDGET INFO) ---")
                print(text[:1000])  # Print first 1000 chars of budget pages
                print("...")
            elif i < 5 or i % 10 == 0:  # Print first 5 pages and every 10th page
                print(f"\n--- PAGE {i+1} ---")
                print(text[:200])  # Print first 200 chars of other pages
                print("...")
        
        print(f"\nPages likely containing budget information: {budget_pages}")
except Exception as e:
    print(f"Error extracting PDF: {str(e)}")
