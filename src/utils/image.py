import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_masks(mask, ax, random_color=False, borders = True):
    """
    Show the mask prediction of the image
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(point_coords, point_labels, ax):
    pos_points = np.array(point_coords)
    
    if pos_points.ndim == 1:
        if pos_points.shape[0] == 2:
            pos_points = pos_points.reshape(1, 2)
    elif pos_points.shape[1] != 2:
        raise ValueError(f"[Error] Shape of point_coords is not valid: {pos_points.shape}")
        
    ax.scatter(pos_points[:, 0], pos_points[:, 1], c='green', s=50)

def show_box(box, ax):
    """
    Show the bounding box of the image
    """
    box = np.array(box)

    if box.ndim == 2 and box.shape == (2, 2):
        box = box.flatten()
    
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))

def show_image_masks_and_prompts(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    """
    Show the image with the masks and the prompts
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i].imshow(image)
        show_masks(mask, axes[i], borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, axes[i])
        if box_coords is not None:
            show_box(box_coords, axes[i])
            pass
        axes[i].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_image_with_masks_and_prompts(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True, save_path='output.png'):
    """
    Save the image with the masks and the prompts
    """

    num_masks = len(masks)
    fig, axes = plt.subplots(1, num_masks, figsize=(5 * num_masks, 8))
    
    if num_masks == 1:
        axes = [axes]
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i].imshow(image)
        show_masks(mask, axes[i], borders=borders)
        
        if point_coords is not None:
            show_points(point_coords, input_labels, axes[i])
            
        if box_coords is not None:
            show_box(box_coords, axes[i])
        
        axes[i].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Info] Image saved to {save_path}")
    plt.close(fig)