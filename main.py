from curses import raw
import json
import os
import numpy as np

from dataset_creator import create_coco_dataset

if __name__ == "__main__":
    
    img_dir = "/Users/thomcope/new-projects/swimming/tracker/data2/m100br_fine2"
    boxes_path = "/Users/thomcope/new-projects/swimming/tracker/data/boxes/m100br_fine2_output_filtered.npy"
    
    img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
    boxes = np.load(boxes_path, allow_pickle=True).tolist()
    
    raw_data = zip(img_paths, boxes)

    custom_info = {
        "description": "Swimmer Dataset v1",
        "version": "1.0",
        "contributor": "tommy",
        "url": "https://test.com/test"
    }
    dataset = create_coco_dataset(raw_data, "swimmer", custom_info)

    # Save to file
    with open("m100br_fine_coco.json", "w") as f:
        json.dump(dataset, f, indent=2)