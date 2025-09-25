from .purchase_order import PurchaseOrderExtractor
from .performa_invoice import PerformaInvoiceExtractor
from .lc_application import LcApplicationExtractor

def get_extractor(doc_type):
    doc_type = doc_type.lower()
    if doc_type == "purchase_order":
        return PurchaseOrderExtractor()
    if doc_type == "performa_invoice":
        return PerformaInvoiceExtractor()
    if doc_type == "lc_application":
        return LcApplicationExtractor()
    return None