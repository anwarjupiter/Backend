import pdfplumber,tempfile,cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path
import numpy as np
import pytesseract
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    template="""
        You are a highly skilled AI assistant trained to extract structured data from unstructured, messy, or raw PDF text. Your task is to intelligently analyze the text and return a meaningful, structured JSON representation of the information it contains.

        ### Instructions:
        1. Carefully read the text extracted from the PDF.
        2. Identify key attributes, fields, or repeated patterns.
        3. Convert them into a clean JSON structure with:
        - Descriptive keys (based on your understanding of the data)
        - Accurate nesting if appropriate
        - Lists for repeated elements
        4. Do **not hallucinate**. If a value is missing or unknown, either skip it or leave it as an empty string or list.
        5. Focus on preserving meaning and hierarchy — create **logical groupings** like `product_details`, `materials`, `prints`, `design_notes`, etc., based on your understanding.

        Respond ONLY with valid, parsable JSON.

        ### Raw PDF Text:
        {rawtext}
        """
)

MODEL_FLASH_2_0 = "gemini-2.0-flash"
GOOGLE_GEMINI_KEY = "AIzaSyCqgpJTOLeA-BIk2lrHw2YojZA37NRBTJo"

llm = ChatGoogleGenerativeAI(
    model=MODEL_FLASH_2_0, 
    temperature=0,
    api_key=GOOGLE_GEMINI_KEY
)

def clean_with_llm(text_chunk):
    mpy = prompt.format(rawtext=text_chunk)
    response = llm.invoke([
        HumanMessage(content=mpy)
    ])
    return response.content

def ocr(image_path):
    text = pytesseract.image_to_string(image_path)
    return text

def deepscan(pdf_path,page_no):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            pages = convert_from_path(pdf_path, dpi=300, output_folder=temp_dir)
            for i, page_img in enumerate(pages):
                if i == page_no:

                    open_cv_image = np.array(page_img) 
                    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR for OpenCV
                    
                    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

                    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    min_area = 5000  # Adjust this threshold based on your needs
                    image_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                    
                    for j, contour in enumerate(image_contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        roi = open_cv_image[y:y+h, x:x+w]
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        if np.var(roi_gray) > 500:  # Variance threshold
                            pil_img = Image.fromarray(roi_gray)
                            return pil_img
        except Exception as e:
            print(f"Deep scan error: {e}")
    
def extract_text_and_images(pdf_path: str, output_img_dir: str):
    text_chunks = []

    Path(output_img_dir).mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in tqdm(enumerate(pdf.pages),desc="Processing PDF",unit="page"):
            text = page.extract_text(layout=True)
            if text:
                text_chunks.append(f"Page {page_num + 1}:\n{text}\n")
                pil_img = deepscan(pdf_path=pdf_path,page_no=page_num)
                ocr_text = ocr(pil_img)
                text_chunks.append(ocr_text+"\n")
            

    with open("PdfExtract/output/1.txt","w") as f:
        f.writelines(text_chunks)
    return ''.join(text_chunks)


if __name__ == "__main__":
    text_chunks = extract_text_and_images(pdf_path="PdfExtract/52J6402_25-04-24.pdf",output_img_dir="PdfExtract/images")
    response = clean_with_llm(text_chunk=text_chunks)
    print(response)