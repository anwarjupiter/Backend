import pdfplumber, csv, os, shutil

def run(pdf_file, output):
    os.makedirs(output, exist_ok=True)

    # Step 1: Extract tables into CSVs
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            if tables:
                for table_index, table in enumerate(tables, start=1):
                    output_file = os.path.join(output, f"page_{page_num}_table_{table_index}.csv")
                    with open(output_file, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        for row in table:
                            writer.writerow(row)
                print(f"✅ Extracted table(s) from page {page_num}")
            else:
                print(f"⚠️ No table found on page {page_num}")

    # Step 2: Zip the output folder
    zip_file_path = shutil.make_archive(output, 'zip', output)
    print(f"✅ All done! Zipped file created at: {zip_file_path}")
    return zip_file_path
