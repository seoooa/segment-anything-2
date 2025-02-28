import autorootcwd
import torch
import cv2
import numpy as np
import click
from tqdm import tqdm

from src.sam2.build_sam import build_sam2_video_predictor
from src.utils.video import show_video_masks, save_video_with_masks, load_video, convert_to_mp4, select_frame_interactively
from src.utils.prompt import get_click_point, get_bounding_box

@click.command()
@click.option("--prompt", default='p', type=click.Choice(["p", "b"]), required=True, help="prompt type (p: click point, b: bounding box)")
@click.option("--video_path", default=None, type=str, help="video file path (if not specified, the file explorer will be opened)")
def main(prompt, video_path):

    if video_path is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        if not video_path:
            raise ValueError("Video file is not selected.")
    
    video_path = convert_to_mp4(video_path)
    output_path = "./output/mask-prediction/" + video_path.split("/")[-1]
    
    # Initialize the predictor
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    input_points, input_labels = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        state = predictor.init_state(video_path)

        # select frame ( default: 0th frame, move with a/d key (0.2s unit))
        selected_frame_idx = select_frame_interactively(video_path, default_frame_idx=0, jump_seconds=0.2)
        print(f"Selected frame: {selected_frame_idx}")

        image = state["images"][selected_frame_idx].cpu().permute(1, 2, 0).numpy()
        image = cv2.resize(image, (original_width, original_height))

        # apply click point or bounding box prompt on the selected frame
        if prompt == "p":
            input_points, input_labels = get_click_point(image, input_points, input_labels)
            frame_idx, object_ids, masks = predictor.add_new_points(inference_state=state, points=input_points, labels=input_labels, frame_idx=selected_frame_idx, obj_id=1)
        elif prompt == "b":
            input_box = get_bounding_box(image)
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state, box=input_box, frame_idx=selected_frame_idx, obj_id=1)

        # propagate for the whole video
        frames = load_video(video_path)
        num_frames = len(frames)
        full_masks = [torch.zeros((original_height, original_width), dtype=torch.uint8) for _ in range(num_frames)]
        
        # perform propagation and fill the predicted mask in the corresponding frame index.
        for current_frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(
                inference_state=state, start_frame_idx=frame_idx, max_frame_num_to_track=None, reverse=False):
            if current_frame_idx < num_frames:
                full_masks[current_frame_idx] = video_res_masks

        show_video_masks(frames, full_masks, 0)
        save_video_with_masks(frames, full_masks, output_path)

if __name__ == "__main__":
    main()