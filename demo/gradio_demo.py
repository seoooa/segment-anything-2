import autorootcwd
import sys
import torch
import cv2
import numpy as np
import gradio as gr
import time

from src.sam2.build_sam import build_sam2_camera_predictor
from src.utils.gradio import show_prompt_on_frame, show_mask_overlay

def inference_stream(point_x, point_y):
    # 모델 초기화
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 카메라를 열 수 없습니다.")
        yield None
        return

    # prompt 초기화: Gradio 입력값을 통해 prompt 정보를 설정.
    prompt = {
        "mode": "point",
        "point_coords": [],
        "point_labels": [],
        "bbox_start": None,
        "bbox_end": None,
        "bbox": None,
        "if_init": False
    }
    # CV2의 마우스 콜백 대신, Gradio UI로부터 전달받은 좌표를 사용
    if point_x is not None and point_y is not None:
        prompt["point_coords"] = [(int(point_x), int(point_y))]
        prompt["point_labels"] = [1]
    
    # 새로운 inference를 위해 일정 프레임(예: 300 프레임)까지만 동작하도록 설정합니다.
    frame_count = 0
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = None, None
            while frame_count < 300:
                ret, frame = cap.read()
                if not ret:
                    break

                # 좌측 출력: 원본 프레임에 prompt overlay 적용
                raw_with_prompt = show_prompt_on_frame(frame, prompt)
                raw_frame_rgb = cv2.cvtColor(raw_with_prompt, cv2.COLOR_BGR2RGB)

                # 모델 추론: 초기 프레임에서 prompt로 tracker 초기화
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
                
                # 우측 출력: mask overlay만 적용 (prompt 미포함)
                if out_mask_logits is not None:
                    inference_frame = show_mask_overlay(frame, out_mask_logits, prompt)
                else:
                    inference_frame = frame.copy()
                inference_frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                
                # 좌측: 원본 + prompt, 우측: mask overlay only
                yield (raw_frame_rgb, inference_frame_rgb)

                time.sleep(0.03)
                frame_count += 1
    finally:
        cap.release()

def main():
    # live 모드를 제거하고, 사용자가 입력 버튼을 누를 때마다 새로운 inference_stream이 실행됨.
    demo = gr.Interface(
        fn=inference_stream,
        title="SAM2 Real-Time Inference",
        description="Display the real-time inference results for the web camera input.",
        inputs=[
            gr.Number(label="X coordinate", value=100),
            gr.Number(label="Y coordinate", value=100)
        ],
        outputs=[
            gr.Image(label="Webcam Input (Prompt Included)"),
            gr.Image(label="Real-Time Inference (Mask Overlay Only)")
        ]
    )
    demo.launch()

if __name__ == "__main__":
    main()