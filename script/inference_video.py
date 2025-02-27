import autorootcwd
import torch
import cv2
import numpy as np
import click
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from src.utils.video import show_video_masks, save_video_with_masks, load_video, convert_to_mp4
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
    output_path = "./output/mask_" + video_path.split("/")[-1]
    
    # Initialize the predictor
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        # Initialize the state of the video
        cap = cv2.VideoCapture(video_path)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        state = predictor.init_state(video_path)

        # Get the first frame of the video
        image = state["images"][0].cpu().permute(1, 2, 0).numpy()
        image = cv2.resize(image, (original_width, original_height))

        # Add the new points or bounding box to the video
        if prompt == "p":
            input_point = get_click_point(image)
            frame_idx, object_ids, masks = predictor.add_new_points(inference_state=state, points=input_point, labels=[1], frame_idx=0, obj_id=1)
        elif prompt == "b":
            input_box = get_bounding_box(image)
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state, box=input_box, frame_idx=0, obj_id=1)

        # Propagate the masks in the video
        all_masks = []
        for current_frame_idx, obj_ids, video_res_masks  in predictor.propagate_in_video(inference_state=state, start_frame_idx=frame_idx, max_frame_num_to_track=None, reverse=False):
            all_masks.append(video_res_masks)

        frames = load_video(video_path)

        show_video_masks(frames, all_masks, 0)
        save_video_with_masks(frames, all_masks, output_path)

if __name__ == "__main__":
    main()