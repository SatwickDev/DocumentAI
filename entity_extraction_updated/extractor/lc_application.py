import os
from pathlib import Path
from pdf2image import convert_from_path
from paddleocr import PPStructureV3
import numpy as np
from bs4 import BeautifulSoup

ENTITY_CODES = [
    "F-40A", "F-23", "F-31C", "F40E", "F31D", "F51A", "F50", "F59", "F32B", "F39A", "F39C",
    "F41A", "F42C", "F42A_2", "F42M", "F42P", "F43P", "F43T", "F44A", "F44E", "F44F", "F44B",
    "F44C", "F44D", "F45A", "F46A", "F47A", "F718", "F48", "F49", "F53A", "F78", "F57A", "F72"
]

def extract_entities_from_html_tables(md_text, existing_entities=None):
    if existing_entities is None:
        existing_entities = {}
    soup = BeautifulSoup(md_text, "html.parser")
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            # Handle merged cells with colspan
            if len(cells) >= 3:
                code_cell = cells[0].get_text(strip=True).replace("-", "").replace(" ", "").lower()
                for code in ENTITY_CODES:
                    code_norm = code.replace("-", "").replace(" ", "").lower()
                    if code_cell == code_norm:
                        if code not in existing_entities:
                            existing_entities[code] = cells[2].get_text(" ", strip=True)
            # Handle possible two-column merged rows with code in colspan=2
            elif len(cells) == 2:
                code_cell = cells[0].get_text(strip=True).replace("-", "").replace(" ", "").lower()
                for code in ENTITY_CODES:
                    code_norm = code.replace("-", "").replace(" ", "").lower()
                    if code_cell == code_norm:
                        if code not in existing_entities:
                            existing_entities[code] = cells[1].get_text(" ", strip=True)
    return existing_entities

class LcApplicationExtractor:
    def __init__(self):
        self.pipeline = PPStructureV3()

    def extract(self, pdf_path, output_dir="output_tables"):
        base_name = Path(pdf_path).stem
        output_subfolder = os.path.join(output_dir, base_name)
        os.makedirs(output_subfolder, exist_ok=True)

        # Step 1: Run OCR and save .md files (filename will be random)
        pages = convert_from_path(pdf_path)
        for page_num, page_img in enumerate(pages, 1):
            result = self.pipeline.predict(input=np.array(page_img))
            for res in result:
                try:
                    res.save_to_markdown(save_path=output_subfolder)
                except Exception:
                    pass
            print(f"Processed page {page_num}")

        # Step 2: After processing, parse all .md files in the output folder using HTML parser
        all_entities = {}
        md_files = [f for f in os.listdir(output_subfolder) if f.endswith('.md')]
        for md_file in md_files:
            md_path = os.path.join(output_subfolder, md_file)
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
                all_entities = extract_entities_from_html_tables(md_text, all_entities)

        return all_entities
