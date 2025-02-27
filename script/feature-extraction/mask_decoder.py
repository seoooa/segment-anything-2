import autorootcwd
import torch
import cv2
import numpy as np
import click

from sam2.build_sam import build_sam2_image_predictor
from src.utils.feature_extraction import apply_pca_and_visualize
from src.utils.prompt import get_click_point

@click.command()
@click.option("--image_path", default=None, type=str, help="image file path (if not specified, the file explorer will be opened)")
def main(image_path):

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


    # initialize the predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_image_predictor(model_cfg, sam2_checkpoint)
    mask_decoder_model = predictor.model.sam_mask_decoder.to(device).eval()
    
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # mask decoder
    hooks = []
    #hooks.append(mask_decoder_model.transformer.layers[0].register_forward_hook(hook_fn("layer1")))
    hooks.append(mask_decoder_model.transformer.layers[1].register_forward_hook(hook_fn("layer2")))
    #hooks.append(mask_decoder_model.transformer.final_attn_token_to_image.register_forward_hook(hook_fn("final_attn_token_to_image")))
    #hooks.append(mask_decoder_model.output_upscaling.register_forward_hook(hook_fn("output_upscaling")))
    #hooks.append(mask_decoder_model.iou_prediction_head.register_forward_hook(hook_fn("iou_prediction_head")))
    #hooks.append(mask_decoder_model.pred_obj_score_head.register_forward_hook(hook_fn("pred_obj_score_head")))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # set point prompt
    input_point = get_click_point(image)
    input_label = np.array([1])

    # Forward Pass
    with torch.no_grad():
        predictor.set_image(image)
        _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

    # apply PCA and save individual plots (save PCA result images to dictionary)
    pca_features = {}
    for name, feature in features.items():
        for idx, item in enumerate(feature):
            for jdx, tensor in enumerate(item):
                print(f"{name}_{idx}_{jdx} shape: {tensor.shape}")
                idx_name = name+f"_{idx}_{jdx}"
                save_path = f"./output/feature-extraction/mask_decoder_self_attention_{idx_name}.jpg"
                if idx == 1:
                    pca_features[idx_name] = apply_pca_and_visualize(tensor, image, "mask_decoder", save_path)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
    main()