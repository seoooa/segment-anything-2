import autorootcwd
import sys
import torch
import cv2 
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open camera. Please check the correct camera index or connection status.")
    sys.exit(1)

# Set the default prompt mode to point ("point" or "box")
prompt_mode = "point"

# point prompt related variables
point_coords = []   # coordinates of clicked points (e.g., [[x, y]])
point_labels = []   # labels of coordinates (1: positive, 0: negative)

# bounding box prompt related variables
bbox_start = None   # starting coordinates of dragging
bbox_end = None     # coordinates during or after dragging
bbox = None         # final bounding box: [[x_min, y_min, x_max, y_max]]

if_init = False     # If a new prompt is input, set to False (tracker initialization)

# Mouse event callback function: different behavior based on prompt mode
def mouse_callback(event, x, y, flags, param):
    global point_coords, point_labels, bbox_start, bbox_end, bbox, if_init, prompt_mode
    if prompt_mode == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            point_coords = [[x, y]]
            point_labels = [1]
            if_init = False
            print(f"New point prompt input: {point_coords}")
    elif prompt_mode == "box":
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_start = (x, y)
            bbox_end = (x, y)
            print(f"Bounding box starting point: {bbox_start}")
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if bbox_start is not None:
                bbox_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if_init = False
            if bbox_start is not None:
                bbox_end = (x, y)
                x_min = min(bbox_start[0], bbox_end[0])
                y_min = min(bbox_start[1], bbox_end[1])
                x_max = max(bbox_start[0], bbox_end[0])
                y_max = max(bbox_start[1], bbox_end[1])
                bbox = [[x_min, y_min, x_max, y_max]]
                print(f"New bounding box prompt input: {bbox}")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out_obj_ids, out_mask_logits = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # If a new prompt is input, perform tracker initialization
        if not if_init:
            if prompt_mode == "point" and len(point_coords) > 0:
                predictor.load_first_frame(frame)
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    0, 1, points=point_coords, labels=point_labels
                )
                if_init = True
            elif prompt_mode == "box" and bbox is not None:
                predictor.load_first_frame(frame)
                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    0, 1, bbox=bbox
                )
                if_init = True
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

        # Display prompt on screen (different drawing based on mode)
        if prompt_mode == "point":
            for pt, label in zip(point_coords, point_labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius=5, color=color, thickness=-1)
        elif prompt_mode == "box":
            if bbox_start is not None and bbox_end is not None:
                x_min = min(bbox_start[0], bbox_end[0])
                y_min = min(bbox_start[1], bbox_end[1])
                x_max = max(bbox_start[0], bbox_end[0])
                y_max = max(bbox_start[1], bbox_end[1])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Mask overlay and center coordinate calculation/display
        if out_mask_logits is not None:
            mask_pred = out_mask_logits
            if isinstance(mask_pred, torch.Tensor):
                mask_pred = mask_pred.cpu().numpy()
            mask_pred = np.squeeze(mask_pred)
            binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255
            if binary_mask.ndim != 2:
                binary_mask = binary_mask.reshape(binary_mask.shape[-2], binary_mask.shape[-1])
            
            alpha = 0.85 
            mask_bool = (binary_mask == 255)
            mask_bool_3c = np.repeat(mask_bool[:, :, None], 3, axis=2)
            dark_frame = (frame * (1.0 - alpha)).astype(np.uint8)
            overlay = np.where(mask_bool_3c, frame, dark_frame)

            # Display prompt on overlay
            if prompt_mode == "point":
                for pt, label in zip(point_coords, point_labels):
                    color = (0, 255, 0) if label == 1 else (0, 0, 255)
                    cv2.circle(overlay, (int(pt[0]), int(pt[1])), radius=5, color=color, thickness=-1)
            elif prompt_mode == "box":
                if bbox_start is not None and bbox_end is not None:
                    x_min = min(bbox_start[0], bbox_end[0])
                    y_min = min(bbox_start[1], bbox_end[1])
                    x_max = max(bbox_start[0], bbox_end[0])
                    y_max = max(bbox_start[1], bbox_end[1])
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

            # Calculate the center of the object (using moments)
            moments = cv2.moments(binary_mask)
            if moments["m00"] != 0:
                cX = int(moments["m10"] / moments["m00"])
                cY = int(moments["m01"] / moments["m00"])
                cv2.circle(overlay, (cX, cY), radius=7, color=(255, 255, 255), thickness=-1)
                cv2.putText(overlay, f"({cX}, {cY})", (cX - 20, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("Overlay", overlay)

        cv2.imshow("Frame", frame)

        # Keyboard input processing
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            prompt_mode = "point"   # Convert to point mode
            point_coords = []
            point_labels = []
            bbox_start = None
            bbox_end = None
            bbox = None
            if_init = False
            print("Point prompt mode activated.")
        elif key == ord("b"):
            prompt_mode = "box"   # Convert to bounding box mode
            point_coords = []
            point_labels = []
            bbox_start = None
            bbox_end = None
            bbox = None
            if_init = False
            print("Bounding box prompt mode activated.")

cap.release()
cv2.destroyAllWindows() 