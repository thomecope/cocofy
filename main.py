import os
import numpy as np
import argparse

from cleaner import main as cleaner
from dataset_creator import main as dataset_creator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='input directory')
    parser.add_argument('--boxes', required=True, help='numpy boxes')
    parser.add_argument('--out_dir', required=False, help='output directory (optional)')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    
    """
    Example:
    --img_dir = "/Users/thomcope/new-projects/swimming/tracker/data2/m100br_fine2"
    --boxes_path = "/Users/thomcope/new-projects/swimming/tracker/notebooks/m100br_fine2_output_0727.npy"
    --out_dir = "/Users/thomcope/new-projects/swimming/tracker/notebooks"
    """
    
    args = parse_args()
        
    img_dir = args.dir
    boxes_path = args.boxes
    output_path = args.out_dir if args.out_dir is not None else os.path.dirname(boxes_path)
    
    filename_parts = os.path.splitext(os.path.basename(boxes_path))
    
    output_boxes = filename_parts[0] + "_cleaned" + filename_parts[1]
    output_boxes = os.path.join(output_path, output_boxes)
    
    output_dataset = filename_parts[0] + "_coco.json"
    output_dataset = os.path.join(output_path, output_dataset) 
    
    boxes = np.load(boxes_path, allow_pickle=True).tolist()
    
    cleaner(img_dir, output_boxes, boxes)
    dataset_creator(img_dir, output_dataset, boxes) 
    
    