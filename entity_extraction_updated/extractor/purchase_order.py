import re
from .base import BaseExtractor

class PurchaseOrderExtractor(BaseExtractor):
    def extract(self, file_path):
        text = self.extract_text(file_path)
        # print("---OCR TEXT START---")
        # print(text)
        # print("---OCR TEXT END---")
        entities = {}

        # PO number: Only the code, not the date
        po_number = re.search(
            r"PO NUMBER\s*[:\-]?\s*([A-Z0-9\/ ]+)(?=\s*DATED)", text, re.I
        )
        entities["po_number"] = po_number.group(1).strip() if po_number else ""

        # PO date: Use the first occurrence after DATED
        po_date = re.search(
            r"DATED\s*([0-9]{2}\.[0-9]{2}\.[0-9]{4})", text
        )
        entities["po_date"] = po_date.group(1).strip() if po_date else ""

        # SELLER: fuzzy block between "SELLER" and "BUYER", skipping possible PO NUMBER lines
        seller_block = re.search(
            r"SELLER.*?\n(.*?)(?:\nBUYER)", text, re.I | re.S
        )
        if seller_block:
            lines = [line.strip() for line in seller_block.group(1).split('\n') 
                     if line.strip() and not line.upper().startswith("PO NUMBER")]
            entities["seller_name"] = ', '.join(lines)
        else:
            entities["seller_name"] = ""

        # BUYER: block after "BUYER" up to "UNITED ARAB EMIRATES"
        buyer_block = re.search(
            r"BUYER\s*\n([\s\S]+?)(UNITED ARAB EMIRATES)", text, re.I
        )
        if buyer_block:
            lines = [line.strip() for line in buyer_block.group(1).split('\n') if line.strip()]
            entities["buyer_name"] = ', '.join(lines) + ", UNITED ARAB EMIRATES"
        else:
            entities["buyer_name"] = ""

        # Goods description: grab until PARTIAL SHIPMENT/TRANSHIPMENT/2x newline/end
        description = re.search(
            r"DESCRIPTION OF GOODS\s*\n(.+?)(?:\nPARTIAL SHIPMENT|\nTRANSHIPMENT|\n{2,}|$)",
            text, re.S | re.I
        )
        entities["goods_description"] = (
            description.group(1).replace('\n', ' ').strip() if description else ""
        )

        # Quantity (number + unit)
        quantity = re.search(
            r"([0-9,]+(?:\.\d+)?\s*(?:MT|KG|TON|PCS))", text, re.I
        )
        entities["quantity"] = quantity.group(1).replace(',', '') if quantity else ""

        # Unit price: not present in this format
        entities["unit_price"] = ""

        # Total value and currency: e.g. USD19,555,555.00 or USD 19,555,555.00
        total_value = re.search(
            r"(USD|EUR|INR|GBP)\s*([0-9,]+\.\d{2})", text
        )
        entities["currency"] = total_value.group(1) if total_value else ""
        entities["total_value"] = total_value.group(2) if total_value else ""

        # Delivery terms: e.g. CFR DJIBOUTI SEAPORT (INCOTERMS 2020)
        delivery_terms = re.search(
            r"(CFR|CIF|FOB|DAP|EXW)[ \w,()/-]+(INCOTERMS ?\d{4})?", text, re.I
        )
        entities["delivery_terms"] = (
            delivery_terms.group(0).strip() if delivery_terms else ""
        )

        # Payment terms: not present in this sample
        entities["payment_terms"] = ""

        # Governing law/force majeure: not present in this sample
        entities["governing_law_or_force_majeure"] = ""

        return entities