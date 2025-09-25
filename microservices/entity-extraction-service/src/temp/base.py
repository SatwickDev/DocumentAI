import pytesseract
from PIL import Image
import os

def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

class BaseExtractor:
    def extract_text(self, file_path):
        if is_pdf(file_path):
            from pdf2image import convert_from_path
            # Convert first page only for now
            images = convert_from_path(file_path, first_page=1, last_page=1)
            img = images[0]
        else:
            img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        return text

    def extract(self, file_path):
        raise NotImplementedError("Subclasses should implement this!")