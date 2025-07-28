import json
from datetime import datetime

def create_coco_dataset(images_data, class_name="object", dataset_info=None):
    """
    Create complete COCO dataset for Roboflow upload.
    
    Args:
        images_data: List of tuples (filename, width, height, bboxes)
                    where bboxes is list of (x1, y1, x2, y2) tuples
        class_name: Name of your object class
        dataset_info: Optional dict with dataset metadata
    
    Returns:
        Dictionary in complete COCO format for Roboflow
    """
    # Default dataset info
    if dataset_info is None:
        dataset_info = {
            "description": f"{class_name} Detection Dataset",
            "version": "1.0",
            "contributor": "Custom Dataset"
        }
    
    current_time = datetime.now().isoformat() + "+00:00"
    
    coco = {
        # Required info section
        "info": {
            "year": str(datetime.now().year),
            "version": dataset_info.get("version", "1.0"),
            "description": dataset_info.get("description", f"{class_name} Detection Dataset"),
            "contributor": dataset_info.get("contributor", "Custom Dataset"),
            "url": dataset_info.get("url", ""),
            "date_created": current_time
        },
        
        # Required licenses section
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "name": "Public Domain"
            }
        ],
        
        # Categories with single class
        "categories": [
            {
                "id": 1,  # Roboflow expects categories to start from 1
                "name": class_name,
                "supercategory": "object"
            }
        ],
        
        "images": [],
        "annotations": []
    }
    
    annotation_id = 1
    
    for image_id, (filename, width, height, bboxes) in enumerate(images_data, 1):
        # Add image with all required fields
        coco["images"].append({
            "id": image_id,
            "license": 1,  # Reference to license
            "file_name": filename,
            "height": height,
            "width": width,
            "date_captured": current_time
        })
        
        # Add annotations with all required fields
        for x1, y1, x2, y2 in bboxes:
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Your single class ID
                "bbox": [x1, y1, w, h],  # [x, y, width, height]
                "area": w * h,
                "segmentation": [],  # Empty for detection only
                "iscrowd": 0  # 0 = individual object, 1 = crowd
            })
            annotation_id += 1
    
    return coco

# Usage examples
if __name__ == "__main__":
    # Example 1: Basic usage
    images = [
        ("image001.jpg", 640, 480, [(100, 50, 300, 400), (350, 100, 500, 350)]),
        ("image002.jpg", 800, 600, [(50, 75, 200, 300)])
    ]
    
    dataset = create_coco_dataset(images, "person")
    
    # Save for Roboflow
    with open("annotations.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset with {len(dataset['images'])} images")
    print(f"Total annotations: {len(dataset['annotations'])}")
    
    # Example 2: With custom metadata
    custom_info = {
        "description": "Custom Person Detection Dataset v2",
        "version": "2.0",
        "contributor": "My Company",
        "url": "https://mycompany.com/dataset"
    }
    
    dataset_v2 = create_coco_dataset(images, "person", custom_info)
    
    with open("annotations_v2.json", "w") as f:
        json.dump(dataset_v2, f, indent=2)