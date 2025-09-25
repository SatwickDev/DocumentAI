import argparse
from extractor import get_extractor

def main():
    parser = argparse.ArgumentParser(description="Document Entity Extractor")
    parser.add_argument("--file", required=True, help="Path to the document image (PNG/JPG/PDF)")
    parser.add_argument("--type", required=True, choices=["purchase_order", "performa_invoice", "lc_application"], help="Document type")
    args = parser.parse_args()

    extractor = get_extractor(args.type)
    if not extractor:
        print(f"No extractor for type: {args.type}")
        return

    result = extractor.extract(args.file)
    print("EXTRACTED ENTITIES:")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()