import json
from datetime import datetime
import cv2
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='input directory')
    parser.add_argument('--boxes', required=True, help='numpy boxes')
    parser.add_argument('--out_dir', required=False, help='output directory (optional)')

    args = parser.parse_args()
    
    return args


def main(img_dir, output_file, boxes):
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    raw_data = zip(img_paths, boxes)
    
    custom_info = {
        "description": "Swimmer Dataset v1",
        "version": "1.0",
        "contributor": "tommy",
        "url": "https://test.com/test"
    }
    dataset = create_coco_dataset(raw_data, "swimmer", custom_info)

    # save to file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)


def create_coco_dataset(images_data, class_name="object", dataset_info=None):
    """
    Create COCO dataset from simple image data.
    
    Args:
        images_data: List of tuples (filename, bboxes) where bboxes is list of (x1, y1, x2, y2) tuples
        class_name: Name of your object class
    
    Returns:
        Dictionary in COCO format
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
                "id": 0,  # Roboflow expects categories to start from 1
                "name": class_name,
                "supercategory": "none"
            }
        ],
        
        "images": [],
        "annotations": []
    }
    
    annotation_id = 0
    
    for image_id, (filename, bboxes) in enumerate(images_data, 0):
        
        img = cv2.imread(filename)
        height, width = img.shape[:2]
        
        coco["images"].append({
            "id": image_id,
            "license": 1,  # Reference to license
            "file_name": os.path.basename(filename),
            "height": height,
            "width": width,
            "date_captured": current_time
        })
        
        # Add annotations
        for box in bboxes:
            x1, y1, x2, y2 = [int(k) for k in box[:4]]
            w, h = x2 - x1, y2 - y1
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "segmentation": [],  # Empty for detection only
                "iscrowd": 0  # 0 = individual object, 1 = crowd
            })
            annotation_id += 1
    
    return coco

if __name__ == "__main__":
    
    """
    Example:
    img_dir = "/Users/thomcope/new-projects/swimming/tracker/data2/m100br_fine2"
    boxes_path = "/Users/thomcope/new-projects/swimming/tracker/notebooks/m100br_fine2_output_0727.npy"
    """
    
    args = parse_args()
        
    img_dir = args.dir
    boxes_path = args.boxes
    output_path = args.out_dir if args.out_dir is not None else os.path.dirname(boxes_path)
       
    filename_parts = os.path.splitext(os.path.basename(boxes_path))
    output_file = filename_parts[0] + "_coco.json"
    output_file = os.path.join(output_path, output_file)  
    
    boxes = np.load(boxes_path, allow_pickle=True).tolist()
    
    main(img_dir, output_file, boxes)