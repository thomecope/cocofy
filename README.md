# cocofy

use this to build image detection dataset

## 🧨 how-to

### 1. generate initial boxes

done elsewhere 😁 returns a `np.array` of boxes

### 2. clean up boxes

use `cleaner.py` to take away bad boxes

### 3. export into coco annotations

use `dataset_creator.py` to export annotations to `json`

## ℹ️ more info

### `cleaner.py`

with a directory of images and saved `numpy` array, go through each image and annotation, edit, and save a cleaned version of anntations. 

when it is run, an `opencv` window will open. use the follow actions: 
```
- click on boxes to delete them (orange boxes become good when clicked)
- ctrl+click and drag to select multiple boxes
- 'del' or 'x': delete selected boxes
- 'j': delete all suspect boxes (yellow ones)
- 'g': mark all suspect boxes as good
- 'n': next image
- 'p': previous image
- 's': save current state
- 'd': delete all boxes in current image
- 'r': reset current image (restore from original)
- 'c': clear selection
- 'q' or esc: quit
```

running example:
```
python cleaner.py --dir images --boxes annotations.npy
```
it will save the cleaned annotations as `<annotations>_cleaned.py`

### `dataset_creator.py`

with a directory of images and saved `numpy` array, export a `json` in COCO format.

running example:
```
python dataset_creator.py --dir images --boxes annotations.npy
```
it will save the `json` file as `<annotations>_coco.json`