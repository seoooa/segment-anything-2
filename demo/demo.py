import torch
import cv2
import numpy as np
import argparse
import autorootcwd
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from src.utils.video import show_video_masks, save_video_with_masks, load_video
from src.utils.prompt import get_click_point, get_bounding_box

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="prompt type")
    args = parser.parse_args()

    # Initialize the predictor
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        video_path = "./data/video/tomato_sample5.mp4"
        output_path = "./output/mask_" + video_path.split("/")[-1]

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
        if args.prompt == "p":
            input_point = get_click_point(image)
            frame_idx, object_ids, masks = predictor.add_new_points(inference_state=state, points=input_point, labels=[1], frame_idx=0, obj_id=1)
        elif args.prompt == "b":
            input_box = get_bounding_box(image)
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state, box=input_box, frame_idx=0, obj_id=1)
        else:
            print("Invalid prompt type (p or b)")
            exit()

        # Propagate the masks in the video
        all_masks = []
        for current_frame_idx, obj_ids, video_res_masks  in predictor.propagate_in_video(inference_state=state, start_frame_idx=frame_idx, max_frame_num_to_track=None, reverse=False):
            all_masks.append(video_res_masks)

        frames = load_video(video_path)

        show_video_masks(frames, all_masks, 0)
        save_video_with_masks(frames, all_masks, output_path)

if __name__ == "__main__":
    main()