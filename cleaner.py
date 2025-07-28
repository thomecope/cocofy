from ast import arg
import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
import json
import copy
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='input directory')
    parser.add_argument('--boxes', required=True, help='numpy boxes')
    parser.add_argument('--out_dir', required=False, help='output directory (optional)')

    args = parser.parse_args()
    
    return args


def main(img_dir, output_file, boxes):
    cleaner = BBoxCleaner(img_dir, output_file)
    cleaner.load_numpy_array(boxes)
    cleaner.run()
    

class BBoxCleaner:
    def __init__(self, images_dir: str, output_file: str):
        self.images_dir = images_dir
        self.output_file = output_file
        self.current_image_idx = 0
        self.original_data = None  # Keep original for reset
        self.data = []  # Working copy - will store: {"filename": str, "boxes": [(x1,y1,x2,y2), ...]}
        self.image = None
        self.display_image = None
        self.window_name = "BBox Cleaner - Click boxes to delete, drag to select multiple, 'n'=next, 'p'=prev, 's'=save, 'q'=quit"
        
        # Colors for visualization
        self.box_color = (0, 255, 0)  # Green - good boxes
        self.hover_color = (0, 0, 255)  # Red - hover
        self.selected_color = (255, 0, 255)  # Magenta - selected
        self.selection_color = (255, 255, 0)  # Cyan - selection rectangle
        self.suspect_color = (0, 255, 255)  # Yellow - suspect boxes (very distinct)
        self.text_color = (255, 255, 255)  # White
        
        # State
        self.mouse_pos = (0, 0)
        self.scale_factor = 1.0
        self.deleted_count = 0
        
        # Selection state
        self.is_selecting = False
        self.selection_start = None
        self.selection_end = None
        self.selected_boxes = set()  # Indices of selected boxes
        
        # Suspect box tracking
        self.deleted_boxes_prev_image = []  # Boxes deleted in previous image
        self.suspect_boxes = set()  # Indices of boxes that overlap with deleted ones
        self.good_boxes = set()  # Boxes manually marked as good (override suspect)
        
    def load_numpy_array(self, box_arrays, image_filenames: Optional[List[str]] = None):
        """
        Load data from numpy arrays format.
        
        Args:
            box_arrays: List of numpy arrays OR numpy object array OR filename
                       Each box is [x1, y1, x2, y2, confidence, class_id]
            image_filenames: Optional list of filenames. If None, generates names
        """
        # Handle case where box_arrays is a filename
        if isinstance(box_arrays, str):
            if box_arrays.endswith('.npy'):
                box_arrays = np.load(box_arrays, allow_pickle=True)
                print(f"Loaded box arrays from .npy file")
            else:
                raise ValueError(f"Unsupported file format: {box_arrays}")
        
        # Convert numpy object array to list if needed
        if isinstance(box_arrays, np.ndarray) and box_arrays.dtype == object:
            box_arrays = [box_arrays[i] for i in range(len(box_arrays))]
        
        # Make deep copy of original data
        self.original_data = copy.deepcopy(box_arrays)
        
        # Convert to working format
        self.data = []
        
        if image_filenames is None:
            # Generate filenames and sort them
            image_files = sorted([f for f in os.listdir(self.images_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            if len(image_files) < len(box_arrays):
                print(f"Warning: Found {len(image_files)} images but {len(box_arrays)} box arrays")
            image_filenames = image_files[:len(box_arrays)]
        
        for i, boxes in enumerate(box_arrays):
            filename = image_filenames[i] if i < len(image_filenames) else f"image_{i:04d}.jpg"
            
            # Extract just the bounding box coordinates (first 4 columns)
            if len(boxes) > 0:
                box_coords = [(float(box[0]), float(box[1]), float(box[2]), float(box[3])) 
                             for box in boxes]
            else:
                box_coords = []
                
            self.data.append({
                "filename": filename,
                "boxes": box_coords
            })
        
        print(f"Loaded {len(self.data)} images with {sum(len(item['boxes']) for item in self.data)} total boxes")
    
    def get_numpy_array(self) -> List[np.ndarray]:
        """
        Convert current data back to numpy array format.
        Returns list of arrays with remaining boxes.
        """
        result = []
        for i, item in enumerate(self.data):
            if len(item['boxes']) > 0:
                # Convert back to numpy array format
                # We don't have confidence/class info anymore, so set to default values
                boxes_array = np.array([
                    [x1, y1, x2, y2, 1.0, 0.0] for x1, y1, x2, y2 in item['boxes']
                ])
            else:
                boxes_array = np.array([]).reshape(0, 6)
            result.append(boxes_array)
        return result
    
    def load_data(self, data: List[Dict[str, Any]]):
        """Load data in format: [{"filename": "img.jpg", "boxes": [(x1,y1,x2,y2), ...]}, ...]"""
        self.original_data = copy.deepcopy(data)
        self.data = copy.deepcopy(data)
        print(f"Loaded {len(self.data)} images with {sum(len(item['boxes']) for item in self.data)} total boxes")
        
    def load_from_coco(self, coco_file: str):
        """Load from COCO JSON file"""
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        # Group annotations by image
        image_data = {}
        for img in coco_data['images']:
            image_data[img['id']] = {
                'filename': img['file_name'],
                'boxes': []
            }
        
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']  # [x, y, w, h]
            # Convert to (x1, y1, x2, y2)
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h
            image_data[img_id]['boxes'].append((x1, y1, x2, y2))
        
        data_list = list(image_data.values())
        self.original_data = copy.deepcopy(data_list)
        self.data = copy.deepcopy(data_list)
        print(f"Loaded from COCO: {len(self.data)} images with {sum(len(item['boxes']) for item in self.data)} total boxes")
    
    def get_display_scale(self, img_height: int, img_width: int, max_height: int = 800) -> float:
        """Calculate scale to fit image on screen"""
        if img_height <= max_height:
            return 1.0
        return max_height / img_height
    
    def scale_boxes(self, boxes: List[Tuple], scale: float) -> List[Tuple]:
        """Scale bounding boxes for display"""
        return [(int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)) 
                for x1, y1, x2, y2 in boxes]
    
    def point_in_box(self, point: Tuple[int, int], box: Tuple[int, int, int, int]) -> bool:
        """Check if point is inside bounding box"""
        px, py = point
        x1, y1, x2, y2 = box
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def boxes_overlap(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float], 
                     threshold: float = 0.3) -> bool:
        """
        Check if two boxes overlap significantly.
        
        Args:
            box1, box2: Bounding boxes as (x1, y1, x2, y2)
            threshold: IoU threshold for considering boxes as overlapping
        
        Returns:
            True if boxes overlap above threshold
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return False  # No intersection
        
        # Calculate areas
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou >= threshold
    
    def find_suspect_boxes(self):
        """Find boxes in current image that overlap with deleted boxes from previous image"""
        if not self.deleted_boxes_prev_image or not self.data or self.current_image_idx >= len(self.data):
            self.suspect_boxes.clear()
            return
        
        current_boxes = self.data[self.current_image_idx]['boxes']
        suspect_indices = set()
        
        for i, current_box in enumerate(current_boxes):
            # Skip if manually marked as good
            if i in self.good_boxes:
                continue
                
            for deleted_box in self.deleted_boxes_prev_image:
                if self.boxes_overlap(current_box, deleted_box):
                    suspect_indices.add(i)
                    break
        
        self.suspect_boxes = suspect_indices
    def boxes_intersect_selection(self, selection_rect: Tuple[int, int, int, int]) -> List[int]:
        """Find boxes that intersect with selection rectangle"""
        if not self.data or self.current_image_idx >= len(self.data):
            return []
            
        sx1, sy1, sx2, sy2 = selection_rect
        # Ensure selection rect is normalized
        sx1, sx2 = min(sx1, sx2), max(sx1, sx2)
        sy1, sy2 = min(sy1, sy2), max(sy1, sy2)
        
        # Scale selection back to original coordinates
        sx1 = int(sx1 / self.scale_factor)
        sy1 = int(sy1 / self.scale_factor)
        sx2 = int(sx2 / self.scale_factor)
        sy2 = int(sy2 / self.scale_factor)
        
        boxes = self.data[self.current_image_idx]['boxes']
        intersecting = []
        
        for i, (bx1, by1, bx2, by2) in enumerate(boxes):
            # Check if boxes intersect
            if not (bx2 < sx1 or bx1 > sx2 or by2 < sy1 or by1 > sy2):
                intersecting.append(i)
        
        return intersecting
    
    def find_clicked_box(self, click_pos: Tuple[int, int]) -> int:
        """Find which box was clicked (returns index, -1 if none)"""
        if not self.data or self.current_image_idx >= len(self.data):
            return -1
            
        # Scale click position back to original coordinates
        orig_x = int(click_pos[0] / self.scale_factor)
        orig_y = int(click_pos[1] / self.scale_factor)
        
        boxes = self.data[self.current_image_idx]['boxes']
        
        # Check from last to first (top box has priority)
        for i in reversed(range(len(boxes))):
            if self.point_in_box((orig_x, orig_y), boxes[i]):
                return i
        return -1
    
    def draw_boxes(self):
        """Draw all bounding boxes on the image"""
        if self.image is None:
            return
            
        self.display_image = self.image.copy()
        
        if not self.data or self.current_image_idx >= len(self.data):
            return
            
        boxes = self.data[self.current_image_idx]['boxes']
        scaled_boxes = self.scale_boxes(boxes, self.scale_factor)
        
        # Draw selection rectangle
        if self.is_selecting and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), self.selection_color, 2)
        
        # Draw boxes
        for i, (x1, y1, x2, y2) in enumerate(scaled_boxes):
            # Determine color based on state
            if i in self.selected_boxes:
                color = self.selected_color
                thickness = 3
            elif i in self.suspect_boxes and i not in self.good_boxes:
                color = self.suspect_color  # Orange for suspect boxes
                thickness = 2
            elif self.point_in_box(self.mouse_pos, (x1, y1, x2, y2)):
                color = self.hover_color
                thickness = 3
            else:
                color = self.box_color
                thickness = 2
            
            # Draw rectangle
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw box index with status indicator
            status = ""
            if i in self.suspect_boxes and i not in self.good_boxes:
                status = "S"  # Suspect
            elif i in self.good_boxes:
                status = "G"  # Good
            
            label = f"{i}{status}"
            cv2.putText(self.display_image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        # Draw info text
        info_text = [
            f"Image {self.current_image_idx + 1}/{len(self.data)}",
            f"Boxes: {len(boxes)} (Good: {len(self.good_boxes)}, Suspect: {len(self.suspect_boxes)})",
            f"Selected: {len(self.selected_boxes)}",
            f"Deleted: {self.deleted_count}",
            f"File: {self.data[self.current_image_idx]['filename']}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(self.display_image, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
    
    def load_current_image(self):
        """Load and display current image"""
        if not self.data or self.current_image_idx >= len(self.data):
            return False
        
        # Store deleted boxes from previous image before clearing
        if hasattr(self, 'pending_deleted_boxes'):
            self.deleted_boxes_prev_image = self.pending_deleted_boxes
            delattr(self, 'pending_deleted_boxes')
        else:
            self.deleted_boxes_prev_image = []
            
        # Clear selection and good boxes when changing images
        # Clear selection and good boxes when changing images
        self.selected_boxes.clear()
        self.good_boxes.clear()
        self.good_boxes.clear()
        
        # Find suspect boxes that overlap with deleted boxes from previous image
        self.find_suspect_boxes()
        
        filename = self.data[self.current_image_idx]['filename']
        img_path = os.path.join(self.images_dir, filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return False
            
        self.image = cv2.imread(img_path)
        if self.image is None:
            print(f"Warning: Could not load image: {img_path}")
            return False
            
        # Calculate display scale
        h, w = self.image.shape[:2]
        self.scale_factor = self.get_display_scale(h, w)
        
        # Resize image for display if needed
        if self.scale_factor != 1.0:
            new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
            self.image = cv2.resize(self.image, (new_w, new_h))
        
        self.draw_boxes()
        return True
    
    def delete_selected_boxes(self):
        """Delete all selected boxes"""
        if not self.selected_boxes:
            return
        
        # Store deleted boxes for next image
        current_boxes = self.data[self.current_image_idx]['boxes']
        deleted_boxes = [current_boxes[i] for i in sorted(self.selected_boxes) 
                        if i < len(current_boxes)]
        
        # Add to pending deleted boxes (will be used for next image)
        if not hasattr(self, 'pending_deleted_boxes'):
            self.pending_deleted_boxes = []
        self.pending_deleted_boxes.extend(deleted_boxes)
            
        # Sort indices in reverse order to delete from end to beginning
        sorted_indices = sorted(self.selected_boxes, reverse=True)
        
        for idx in sorted_indices:
            if idx < len(self.data[self.current_image_idx]['boxes']):
                deleted_box = self.data[self.current_image_idx]['boxes'].pop(idx)
                self.deleted_count += 1
                print(f"Deleted box {idx}: {deleted_box}")
        
        # Update good_boxes and suspect_boxes indices after deletion
        self._update_box_indices_after_deletion(sorted_indices)
        
        self.selected_boxes.clear()
        self.draw_boxes()
    
    def _update_box_indices_after_deletion(self, deleted_indices: List[int]):
        """Update good_boxes and suspect_boxes indices after boxes are deleted"""
        # Sort deleted indices
        deleted_sorted = sorted(deleted_indices)
        
        # Update good_boxes
        new_good_boxes = set()
        for idx in self.good_boxes:
            # Count how many deleted indices are before this index
            offset = sum(1 for d_idx in deleted_sorted if d_idx < idx)
            new_idx = idx - offset
            if new_idx >= 0:  # Only keep if still valid
                new_good_boxes.add(new_idx)
        self.good_boxes = new_good_boxes
        
        # Update suspect_boxes
        new_suspect_boxes = set()
        for idx in self.suspect_boxes:
            if idx not in deleted_indices:  # Only if not deleted
                offset = sum(1 for d_idx in deleted_sorted if d_idx < idx)
                new_idx = idx - offset
                if new_idx >= 0:
                    new_suspect_boxes.add(new_idx)
    def delete_single_box(self, box_idx: int):
        """Delete a single box and track it for next image"""
        if box_idx < 0 or box_idx >= len(self.data[self.current_image_idx]['boxes']):
            return
            
        # Store deleted box for next image
        deleted_box = self.data[self.current_image_idx]['boxes'][box_idx]
        if not hasattr(self, 'pending_deleted_boxes'):
            self.pending_deleted_boxes = []
        self.pending_deleted_boxes.append(deleted_box)
        
        # Remove the box
        self.data[self.current_image_idx]['boxes'].pop(box_idx)
        self.deleted_count += 1
        print(f"Deleted box {box_idx}: {deleted_box}")
        
        # Update indices after deletion
        self._update_box_indices_after_deletion([box_idx])
    
    def delete_all_suspect_boxes(self):
        """Delete all suspect boxes (yellow ones)"""
        if not self.suspect_boxes:
            print("No suspect boxes to delete")
            return
        
        # Only delete suspect boxes that haven't been marked as good
        suspect_to_delete = [i for i in self.suspect_boxes if i not in self.good_boxes]
        
        if not suspect_to_delete:
            print("All suspect boxes have been marked as good")
            return
            
        # Get current boxes and store the ones we're deleting
        current_boxes = self.data[self.current_image_idx]['boxes'].copy()
        deleted_boxes = [current_boxes[i] for i in suspect_to_delete 
                        if i < len(current_boxes)]
        
        # Add to pending deleted boxes
        if not hasattr(self, 'pending_deleted_boxes'):
            self.pending_deleted_boxes = []
        self.pending_deleted_boxes.extend(deleted_boxes)
        
        print(f"Deleting {len(suspect_to_delete)} suspect boxes: {sorted(suspect_to_delete)}")
        
        # Create new box list without the deleted ones
        new_boxes = []
        for i, box in enumerate(current_boxes):
            if i not in suspect_to_delete:
                new_boxes.append(box)
        
        self.data[self.current_image_idx]['boxes'] = new_boxes
        self.deleted_count += len(suspect_to_delete)
        
        print(f"Successfully deleted {len(suspect_to_delete)} suspect boxes")
        
        # Completely rebuild tracking sets instead of trying to update indices
        self._rebuild_tracking_sets_after_deletion(suspect_to_delete)
        self.draw_boxes()
    
    def _rebuild_tracking_sets_after_deletion(self, deleted_indices: List[int]):
        """Rebuild good_boxes and suspect_boxes sets from scratch after deletion"""
        # Clear all tracking - safer than trying to update indices
        old_good = self.good_boxes.copy()
        old_suspect = self.suspect_boxes.copy()
        
        self.good_boxes.clear()
        self.suspect_boxes.clear()
        
        print(f"Rebuilt tracking sets after deleting {sorted(deleted_indices)}")
        print(f"Cleared {len(old_good)} good boxes and {len(old_suspect)} suspect boxes")
        
        # Re-run suspect detection from scratch
        self.find_suspect_boxes()
    
    def mark_box_as_good(self, box_idx: int):
        """Mark a suspect box as good (overrides suspect status)"""
        if box_idx >= 0 and box_idx < len(self.data[self.current_image_idx]['boxes']):
            self.good_boxes.add(box_idx)
            print(f"Marked box {box_idx} as good")
            self.draw_boxes()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Ctrl+click starts selection
                self.is_selecting = True
                self.selection_start = (x, y)
                self.selection_end = (x, y)
            else:
                # Regular click - find clicked box
                box_idx = self.find_clicked_box((x, y))
                if box_idx >= 0:
                    # Check if it's a suspect box
                    if box_idx in self.suspect_boxes and box_idx not in self.good_boxes:
                        # Clicking on suspect box marks it as good
                        self.mark_box_as_good(box_idx)
                    else:
                        # Regular deletion
                        self.delete_single_box(box_idx)
                        # Clear selection since indices may have changed
                        self.selected_boxes.clear()
                        self.draw_boxes()
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_selecting and self.selection_start:
                self.selection_end = (x, y)
                self.draw_boxes()
            else:
                # Regular mouse move - just redraw for hover effect
                self.draw_boxes()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_selecting:
                # Finished selection
                if self.selection_start and self.selection_end:
                    selection_rect = (*self.selection_start, *self.selection_end)
                    intersecting = self.boxes_intersect_selection(selection_rect)
                    self.selected_boxes.update(intersecting)
                    print(f"Selected {len(intersecting)} boxes")
                
                self.is_selecting = False
                self.selection_start = None
                self.selection_end = None
                self.draw_boxes()
    
    def next_image(self):
        """Go to next image"""
        if self.current_image_idx < len(self.data) - 1:
            self.current_image_idx += 1
            self.load_current_image()
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_current_image()
    
    def reset_current_image(self):
        """Reset current image to original state"""
        if self.original_data and self.current_image_idx < len(self.original_data):
            if hasattr(self.original_data[self.current_image_idx], '__iter__'):
                # Handle numpy array format
                if len(self.original_data[self.current_image_idx]) > 0:
                    original_boxes = [(float(box[0]), float(box[1]), float(box[2]), float(box[3])) 
                                    for box in self.original_data[self.current_image_idx]]
                else:
                    original_boxes = []
            else:
                # Handle dict format
                original_boxes = copy.deepcopy(self.original_data[self.current_image_idx]['boxes'])
            
            old_count = len(self.data[self.current_image_idx]['boxes'])
            self.data[self.current_image_idx]['boxes'] = original_boxes
            new_count = len(original_boxes)
            restored = new_count - old_count
            if restored > 0:
                self.deleted_count = max(0, self.deleted_count - restored)
            
            self.selected_boxes.clear()
            self.good_boxes.clear()
            print(f"Reset current image: restored {restored} boxes")
            self.find_suspect_boxes()  # Recalculate suspect boxes
            self.draw_boxes()
    
    def save_data(self, output_file: str):
        """Save cleaned data back to file"""
        output = self.get_numpy_array()
        
        # Use numpy object array to handle ragged arrays
        output_np = np.empty(len(output), object)
        for i, arr in enumerate(output):
            output_np[i] = arr
        
        # Save as .npy file
        if not output_file.endswith('.npy'):
            output_file += '.npy'
            
        np.save(output_file, output_np)
        
        remaining_boxes = sum(len(item['boxes']) for item in self.data)
        print(f"Saved to {output_file}. Remaining boxes: {remaining_boxes}, Deleted: {self.deleted_count}")
    
    def run(self):
        """Main loop"""
        if not self.data:
            print("No data loaded!")
            return
            
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.load_current_image()
        
        print("\nControls:")
        print("- Click on boxes to delete them (orange boxes become good when clicked)")
        print("- Ctrl+Click and drag to select multiple boxes")
        print("- 'Del' or 'x': Delete selected boxes")
        print("- 'j': Delete all suspect boxes (yellow ones)")
        print("- 'g': Mark all suspect boxes as good")
        print("- 'n' or Right Arrow: Next image")
        print("- 'p' or Left Arrow: Previous image") 
        print("- 's': Save current state")
        print("- 'd': Delete all boxes in current image")
        print("- 'r': Reset current image (restore from original)")
        print("- 'c': Clear selection")
        print("- 'q' or ESC: Quit")
        print("\nBox Colors:")
        print("- Green: Normal boxes")
        print("- Yellow: Suspect boxes (overlap with deleted boxes from previous image)")
        print("- Red: Hover highlight")
        print("- Magenta: Selected boxes")
        
        while True:
            if self.display_image is not None:
                cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('n') or key == 83:  # Right arrow
                self.next_image()
            elif key == ord('p') or key == 81:  # Left arrow
                self.prev_image()
            elif key == ord('s'):
                self.save_data(self.output_file)
            elif key == ord('d'):
                # Delete all boxes in current image
                count = len(self.data[self.current_image_idx]['boxes'])
                
                # Store all current boxes as deleted for next image
                if not hasattr(self, 'pending_deleted_boxes'):
                    self.pending_deleted_boxes = []
                self.pending_deleted_boxes.extend(self.data[self.current_image_idx]['boxes'])
                
                self.data[self.current_image_idx]['boxes'] = []
                self.deleted_count += count
                self.selected_boxes.clear()
                self.good_boxes.clear()
                self.suspect_boxes.clear()
                print(f"Deleted all {count} boxes from current image")
                self.draw_boxes()
            elif key == ord('j'):
                self.delete_all_suspect_boxes()
            elif key == ord('r'):
                self.reset_current_image()
            elif key == ord('c'):
                self.selected_boxes.clear()
                print("Cleared selection")
                self.draw_boxes()
            elif key == ord('g'):
                # Mark all suspect boxes as good
                if self.suspect_boxes:
                    self.good_boxes.update(self.suspect_boxes)
                    print(f"Marked {len(self.suspect_boxes)} suspect boxes as good")
                    self.draw_boxes()
            elif key == ord('x') or key == 255:  # Delete key
                self.delete_selected_boxes()
        
        cv2.destroyAllWindows()


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
    output_file = filename_parts[0] + "_cleaned" + filename_parts[1]
    output_file = os.path.join(output_path, output_file)
    
    boxes = np.load(boxes_path, allow_pickle=True).tolist()
    
    main(img_dir, output_file, boxes) 