import autorootcwd
import torch
import cv2
import numpy as np
import click
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.utils.prompt import get_click_point, get_bounding_box
from src.utils.image import show_image_masks_and_prompts, save_image_with_masks_and_prompts

@click.command()
@click.option("--prompt", default='p', type=click.Choice(["p", "b"]), required=True, help="prompt type (p: click point, b: bounding box)")
@click.option("--image_path", default=None, type=str, help="image file path (if not specified, the file explorer will be opened)")
def main(prompt, image_path):
    
    if image_path is None:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
        if not image_path:
            raise ValueError("Image file is not selected.")
    output_path = "./output/mask_" + image_path.split("/")[-1]

    # load the image and convert to RGB (the image is in BGR format)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image file is not loaded.")
    original_height, original_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # initialize the predictor
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")
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

if __name__ == "__main__":
    main()