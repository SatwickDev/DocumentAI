from training_dataset import datasets

# Check dataset structure
category = "Purchase Order"
po_data = datasets[category]
print("Category:", category)
print("Data type:", type(po_data))
print("Length:", len(po_data))
print("First item type:", type(po_data[0]))
print("First item:", po_data[0])

# Check if it's a list of strings or dictionaries
if isinstance(po_data[0], dict):
    print("Keys in first item:", po_data[0].keys())