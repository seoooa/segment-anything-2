import autorootcwd
import torch
import cv2
import click

from sam2.build_sam import build_sam2_image_predictor
from src.utils.feature_extraction import preprocess_image, apply_pca_and_visualize, plot_combined_features

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
    image_encoder_model = predictor.model.image_encoder.to(device).eval()
    
    # feature extraction with hooks
    features = {}
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # image encoder
    hooks = []
    hooks.append(image_encoder_model.trunk.patch_embed.register_forward_hook(hook_fn("patch_embed")))
    hooks.append(image_encoder_model.trunk.blocks[1].register_forward_hook(hook_fn("scale1")))  # 144 channels
    hooks.append(image_encoder_model.trunk.blocks[7].register_forward_hook(hook_fn("scale2")))  # 288 channels
    hooks.append(image_encoder_model.trunk.blocks[43].register_forward_hook(hook_fn("scale3")))  # 576 channels
    hooks.append(image_encoder_model.trunk.blocks[47].register_forward_hook(hook_fn("scale4")))  # 1152 channels
    hooks.append(image_encoder_model.neck.register_forward_hook(hook_fn("fpn")))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = preprocess_image(image_path, device)

    # Forward Pass
    with torch.no_grad():
        _ = image_encoder_model(inputs["pixel_values"])

    # apply PCA
    pca_features = {}
    for name, feature in features.items():
        if "fpn" in name:
            for idx, item in enumerate(feature):
                for jdx, tensor in enumerate(item):
                    print(f"{name}_{idx}_{jdx} shape: {tensor.shape}")
                    if tensor.ndim == 4:
                        tensor = tensor.permute(0, 2, 3, 1) # (b, c, h, w) -> (b, h, w, c)
                    idx_name = name+f"_{idx}_{jdx}"
                    pca_features[idx_name] = apply_pca_and_visualize(tensor, image, "image_encoder")
        else :
            print(name, feature.shape)
            pca_features[name] = apply_pca_and_visualize(feature, image, "image_encoder")
    
    non_fpn_features = {name: img for name, img in pca_features.items() if "fpn" not in name}
    plot_combined_features(image, non_fpn_features, "./output/feature-extraction/image_encoder_before_FPN.jpg")

    fpn_features = {name: img for name, img in pca_features.items() if "fpn" in name}
    plot_combined_features(image, fpn_features, "./output/feature-extraction/image_encoder_FPN.jpg")
    
    fpn_features_local = {name: img for name, img in pca_features.items() if "fpn_0" in name}
    plot_combined_features(image, fpn_features_local, "./output/feature-extraction/image_encoder_FPN_Local.jpg")

    fpn_features_global = {name: img for name, img in pca_features.items() if "fpn_1" in name}
    plot_combined_features(image, fpn_features_global, "./output/feature-extraction/image_encoder_FPN_Global.jpg")

    # Remove hooks
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
    main()