import cv2
import numpy as np
import torch

def show_prompt_on_frame(frame, prompt):
    """
    Gradio용: 프롬프트 정보를 이미지(frame) 위에 오버레이하는 함수.
    
    Args:
        frame (numpy.ndarray): 원본 이미지.
        prompt (dict): 'point' 또는 'box' 모드에 따른 프롬프트 정보. 
            예를 들어, 'point' 모드이면 "point_coords"에 좌표 리스트가, 
            'box' 모드이면 "bbox", "bbox_start", "bbox_end" 등이 포함됨.
    
    Returns:
        frame_display (numpy.ndarray): 프롬프트 정보가 그려진 이미지.
    """
    frame_display = frame.copy()
    
    if prompt.get("mode") == "point":
        for pt in prompt.get("point_coords", []):
            cv2.circle(frame_display, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 0), thickness=-1)
    elif prompt.get("mode") == "box":
        if prompt.get("bbox") is not None:
            bbox = prompt["bbox"]
            if isinstance(bbox, list) and len(bbox) > 0:
                # bbox가 리스트의 형태로 들어왔다고 가정 (예: [[x_min, y_min, x_max, y_max]])
                x_min, y_min, x_max, y_max = bbox[0]
                cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        elif prompt.get("bbox_start") is not None and prompt.get("bbox_end") is not None:
            start = prompt["bbox_start"]
            end = prompt["bbox_end"]
            x_min = int(min(start[0], end[0]))
            y_min = int(min(start[1], end[1]))
            x_max = int(max(start[0], end[0]))
            y_max = int(max(start[1], end[1]))
            cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    
    return frame_display

def show_mask_overlay(frame, out_mask_logits, prompt):
    """
    Gradio용: 모델의 mask logits를 기반으로 mask overlay를 생성하고, 
    프롬프트 정보도 함께 이미지에 표시하는 함수.
    
    Args:
        frame (numpy.ndarray): 현재 프레임 이미지.
        out_mask_logits (torch.Tensor or numpy.ndarray): 모델의 mask logits.
        prompt (dict): 프롬프트 관련 정보.
    
    Returns:
        overlay (numpy.ndarray): mask overlay와 프롬프트 정보가 적용된 이미지.
    """
    # mask logits가 torch.Tensor면 numpy 배열로 변환
    if isinstance(out_mask_logits, torch.Tensor):
        mask_pred = out_mask_logits.cpu().numpy()
    else:
        mask_pred = out_mask_logits
        
    mask_pred = np.squeeze(mask_pred)
    binary_mask = (mask_pred > 0.5).astype(np.uint8) * 255
    if binary_mask.ndim != 2:
        binary_mask = binary_mask.reshape(binary_mask.shape[-2], binary_mask.shape[-1])
    
    alpha = 0.85
    dark_frame = (frame * (1.0 - alpha)).astype(np.uint8)
    mask_bool = (binary_mask == 255)
    mask_bool_3c = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
    
    # mask 영역은 원본, 나머지는 어둡게 처리한 overlay 생성
    overlay = np.where(mask_bool_3c, frame, dark_frame)
    
    # 프롬프트 정보도 다시 overlay에 그리기 (필요 시)
    if prompt.get("mode") == "point":
        for pt in prompt.get("point_coords", []):
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), radius=5, color=(0, 255, 0), thickness=-1)
    elif prompt.get("mode") == "box" and prompt.get("bbox") is not None:
        bbox = prompt["bbox"]
        if isinstance(bbox, list) and len(bbox) > 0:
            x_min, y_min, x_max, y_max = bbox[0]
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
    
    # 모멘트를 이용하여 mask의 중심 계산 후 표시
    moments = cv2.moments(binary_mask)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
        cv2.circle(overlay, (cX, cY), radius=7, color=(255, 255, 255), thickness=-1)
        cv2.putText(overlay, f"({cX}, {cY})", (cX - 20, cY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay