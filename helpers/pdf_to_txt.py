# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader(file_path="input/Comments.pdf",extraction_mode="plain")
# documents = loader.load()

# output = ""
# for doc in documents:
#     output += doc.page_content

# with open("output/comments.txt","w") as file:
#     file.write(output)

# print("Resume Text file created !")


# import pdfplumber

# output_tables = []

# with pdfplumber.open("input/tn.pdf") as pdf:
#     for i, page in enumerate(pdf.pages):
#         tables = page.extract_tables()
#         for table in tables:
#             output_tables.append(table)

# # Save to CSV or TXT
# import csv
# with open("output/tn.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     for table in output_tables:
#         for row in table:
#             writer.writerow(row)

# print("Table data extracted and saved!")


import pdfplumber
import csv
import os

# Input PDF
pdf_path = "input/tn.pdf"
output_folder = "output/tables"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load PDF and extract tables
with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        if tables:
            for table_index, table in enumerate(tables, start=1):
                output_file = os.path.join(output_folder, f"page_{page_num}_table_{table_index}.csv")
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for row in table:
                        writer.writerow(row)
            print(f"✅ Extracted table(s) from page {page_num}")
        else:
            print(f"⚠️ No table found on page {page_num}")

print("✅ All done! Check the 'output/tables' folder.")
