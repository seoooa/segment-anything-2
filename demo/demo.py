import autorootcwd
import torch
import cv2
import numpy as np
import sys
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm
import click

from src.sam2.build_sam import build_sam2_image_predictor, build_sam2_video_predictor, build_sam2_camera_predictor
from src.utils.prompt import get_click_point, get_bounding_box, mouse_callback_real_time, process_keyboard_input
from src.utils.image import show_image_masks_and_prompts, save_image_with_masks_and_prompts
from src.utils.video import show_video_masks, save_video_with_masks, load_video
from src.utils.real_time import show_prompt_on_frame, show_mask_overlay

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

def demo_image(prompt, image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
            raise ValueError("Image file is not loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor = build_sam2_image_predictor(model_cfg, sam2_checkpoint)
    predictor.set_image(image)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if prompt == "p":
            input_point = get_click_point(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels= np.array([1]),
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            show_image_masks_and_prompts(image, masks, scores, point_coords=input_point, input_labels=[1], borders=True)
            save_image_with_masks_and_prompts(image, masks, scores, point_coords=input_point, input_labels=[1], borders=True, save_path=output_path)

        elif prompt == "b":
            input_box = get_bounding_box(image)
            masks, scores, logits = predictor.predict(
                box=input_box,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            show_image_masks_and_prompts(image, masks, scores, box_coords=input_box, borders=True)  
            save_image_with_masks_and_prompts(image, masks, scores, box_coords=input_box, borders=True, save_path=output_path)

def demo_video(prompt, video_path, output_path):
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

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

def demo_real_time():
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    
    # initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera. Please check the correct camera index or connection status.")
        sys.exit(1)

    # initialize prompt
    prompt = {
        "mode": "point",
        "point_coords": [],
        "point_labels": [],
        "bbox_start": None,
        "bbox_end": None,
        "bbox": None,
        "if_init": False
    }
    
    # initialize window
    cv2.namedWindow("Real-Time Camera")
    cv2.setMouseCallback("Real-Time Camera", mouse_callback_real_time, prompt)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_obj_ids, out_mask_logits = None, None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # get key input and update prompt values via function call
            if process_keyboard_input(prompt):
                break
            # if new prompt is input, initialize tracker
            if not prompt["if_init"]:
                if prompt["mode"] == "point" and prompt["point_coords"]:
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        0, 1, points=prompt["point_coords"], labels=prompt["point_labels"]
                    )
                    prompt["if_init"] = True
                elif prompt["mode"] == "box" and prompt["bbox"] is not None:
                    predictor.load_first_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        0, 1, bbox=prompt["bbox"]
                    )
                    prompt["if_init"] = True
            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)
            
            show_prompt_on_frame(frame, prompt) # draw prompt information on the original frame

            if out_mask_logits is not None:
                show_mask_overlay(frame, out_mask_logits, prompt) # show mask overlay (show center information of the object)
        
    cap.release()
    cv2.destroyAllWindows()



@click.command()
@click.option("--prompt", default='p', type=click.Choice(["p", "b"]), required=True, help="prompt type (p: click point, b: bounding box)")
@click.option("--input", type=click.Choice(["image", "video", "real-time"]), required=True, help="input type(image, video, real-time)")
def main(input, prompt):

    # Initialize the predictor
    if input == "image":
        image_path = "./data/demo_image.jpg"
        output_path = "./output/mask-prediction/" + image_path.split("/")[-1]
        demo_image(prompt, image_path, output_path)
        
    elif input == "video":
        video_path = "./data/demo_video.mp4"
        output_path = "./output/mask-prediction/" + video_path.split("/")[-1]
        demo_video(prompt, video_path, output_path)
        
    elif input == "real-time":
        demo_real_time()

if __name__ == "__main__":
    main()