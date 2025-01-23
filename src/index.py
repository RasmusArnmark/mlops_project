from torchvision.datasets import ImageFolder

def verify_class_mapping(dataset_path, class_mapping):
    dataset = ImageFolder(dataset_path)
    folder_classes = dataset.classes
    index_to_class = {idx: label for idx, label in enumerate(folder_classes)}

    # Check for mismatches and print correct mapping
    print("Verifying class mappings...")
    for idx, label in index_to_class.items():
        mapped_label = class_mapping.get(idx)
        if mapped_label != label:
            print(f"Mismatch at index {idx}: Expected '{mapped_label}', Found '{label}'")
        else:
            print(f"Correct mapping for index {idx}: '{label}'")
    
    print("\nCorrect mapping based on detected folder structure:")
    correct_mapping = {idx: label for idx, label in enumerate(folder_classes)}
    for idx, label in correct_mapping.items():
        print(f"{idx}: '{label}'")
    
    return correct_mapping

# Example usage
if __name__ == "__main__":
    dataset_path = "data/processed/train"  # Path to the train folder
    from src.model_handler import CLASS_MAPPING  # Import CLASS_MAPPING
    correct_mapping = verify_class_mapping(dataset_path, CLASS_MAPPING)
