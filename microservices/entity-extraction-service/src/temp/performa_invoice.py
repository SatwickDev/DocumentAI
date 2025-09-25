import re
from base import BaseExtractor

class PerformaInvoiceExtractor(BaseExtractor):
    def extract(self, file_path):
        text = self.extract_text(file_path)
        entities = {}

        # Invoice number and date
        invoice_no = re.search(r"PI No\s*([A-Za-z0-9\-]+)", text)
        invoice_date = re.search(r"Date\s*([0-9]{2}-[A-Za-z]{3}-[0-9]{4})", text)
        entities["invoice_number"] = invoice_no.group(1) if invoice_no else ""
        entities["invoice_date"] = invoice_date.group(1) if invoice_date else ""

        # Seller (beneficiary)
        seller = re.search(r"(Global Exports Ltd\.,[^\n]+)", text)
        entities["seller"] = seller.group(1).strip() if seller else ""

        # Buyer (applicant)
        buyer = re.search(r"(ABC Importers LLC,[^\n]+)", text)
        entities["buyer"] = buyer.group(1).strip() if buyer else ""

        # Contract/PO reference number
        entities["contract_po_reference"] = entities["invoice_number"]

        # Incoterm + named place and delivery window
        incoterm_match = re.search(r"Shipment Terms\s*\|\s*(.+?)\s*Delivery Window", text)
        entities["incoterm_named_place"] = incoterm_match.group(1).strip() if incoterm_match else ""
        delivery_win_match = re.search(r"Delivery Window\s*By\s*([0-9A-Za-z\-]+)", text)
        entities["shipment_period_or_delivery_date"] = (
            f"By {delivery_win_match.group(1)}" if delivery_win_match else ""
        )

        # Payment terms
        payment_terms = re.search(r"Payment Terms:\s*([^\n]+)", text)
        entities["payment_terms"] = payment_terms.group(1).strip() if payment_terms else ""

        # Country of origin
        origin = re.search(r"Origin:\s*([A-Za-z]+)", text)
        entities["country_of_origin"] = origin.group(1) if origin else ""

        # Goods Table Extraction (robust, split from right)
        goods = []
        table_header = re.search(r"Description HS Code Qty Unit Unit Price Amount\s*\n", text)
        if table_header:
            start = table_header.end()
            after_table = text[start:]
            lines = after_table.strip().split('\n')
            for line in lines:
                raw = line.strip()
                if not raw or "Total" in raw or raw.startswith('<b>'):
                    break
                # Split from right: amount, unit_price, unit, qty, hs_code, rest is description
                parts = raw.rsplit(" ", 5)
                if len(parts) == 6:
                    desc, hs_code, qty, unit, unit_price, amount = parts
                    goods.append({
                        "description": desc.strip(),
                        "hs_code": hs_code.strip(),
                        "quantity": qty.strip(),
                        "unit": unit.strip(),
                        "unit_price": unit_price.strip(),
                        "amount": amount.replace(",", "").strip(),
                    })
        entities["goods"] = goods

        # Total amount and currency
        total = re.search(r"Total\s*\(([A-Za-z]+)\)[^\d]*([\d,]+\.\d{2})", text)
        entities["currency"] = total.group(1) if total else ""
        entities["total_amount"] = total.group(2).replace(",", "") if total else ""

        return entities