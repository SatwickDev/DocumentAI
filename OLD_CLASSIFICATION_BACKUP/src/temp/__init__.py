from .purchase_order import PurchaseOrderExtractor
from .performa_invoice import PerformaInvoiceExtractor


def get_extractor(doc_type):
    if doc_type == "purchase_order":
        return PurchaseOrderExtractor()
    if doc_type == "performa_invoice":
        return PerformaInvoiceExtractor()
    return None
