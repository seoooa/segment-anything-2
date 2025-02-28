import autorootcwd
import torch
import cv2
import numpy as np
import ffmpeg

def load_video(video_path):
    """
    Load the video from the video path
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    return frames

def convert_to_mp4(input_video_path):
    """
    Convert the video file to mp4 format using ffmpeg module.
    
    input_video_path: path to the original video file to be converted
    """
    # check if the input video file is already mp4 file
    if input_video_path.lower().endswith('.mp4'):
        return input_video_path

    # generate the output video path: change the extension to _converted.mp4
    output_video_path = input_video_path.rsplit('.', 1)[0] + '.mp4'
    try:
        (
            ffmpeg
            .input(input_video_path)
            .output(output_video_path,
                    vcodec='libx264', preset='fast', crf=22,
                    acodec='aac', strict='-2')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"[Info] Converted video is saved to {output_video_path}")
        return output_video_path
    except ffmpeg.Error as e:
        print("[Error] Error occurred during ffmpeg processing:")
        print(e.stderr.decode('utf8'))
        return None

def show_video_masks(frames, masks, start_frame_idx):
    """
    Show the video with the masks (with center of the mask)

    frames: list of frames
    masks: list of masks
    start_frame_idx: start frame index

    """
    original_height, original_width = frames[0].shape[:2]

    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        if mask is None:
            cv2.imshow('Video with Masks', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            # process the mask
            mask = mask.cpu().numpy().squeeze()
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

            mask_binary = (mask > 0).astype(np.float32)
            mask_binary = mask_binary[:, :, np.newaxis]
            
            mask_binary = np.where(mask_binary > 0, 1.0, 0.1)
            frame_segmented = (frame * mask_binary).astype(np.uint8)
            
            # Calculate the center of the mask
            M = cv2.moments((mask_binary[:,:,0] > 0.5).astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                frame_segmented = cv2.circle(frame_segmented, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(frame_segmented, f"({cX}, {cY})", (cX + 10, cY + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

            cv2.imshow('Video with Masks', cv2.cvtColor(frame_segmented, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(50) & 0xFF == ord('q'): # wait for 50ms and check if 'q' is pressed
            break

    cv2.destroyAllWindows()

def save_video_with_masks(frames, masks, save_path):
    """
    Save the video with the masks

    frames: list of frames
    masks: list of masks
    save_path: path to save the video

    """
    if len(frames) == 0 or masks is None:
        print("[Warning] No frames or masks to save")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
    
    all_masks = [None] * len(frames)
    for i, mask in enumerate(masks):
        if mask is not None:
            all_masks[i] = mask
    
    for frame, mask in zip(frames, all_masks):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().permute(1, 2, 0).numpy()
        if mask is None:
            frame_to_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # process the mask
            mask = mask.cpu().numpy().squeeze()
            mask_binary = (mask > 0).astype(np.float32)
            mask_binary = mask_binary[:, :, np.newaxis]
            
            if frame.shape[:2] != mask_binary.shape[:2]:
                frame = cv2.resize(frame, (mask_binary.shape[1], mask_binary.shape[0]))
            
            mask_binary = np.where(mask_binary > 0, 1.0, 0.1)
            frame_segmented = (frame * mask_binary).astype(np.uint8)
            
            # Calculate the center of the mask
            M = cv2.moments((mask_binary[:,:,0] > 0.5).astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                frame_segmented = cv2.circle(frame_segmented, (cX, cY), 5, (0, 0, 255), -1)
                cv2.putText(frame_segmented, f"({cX}, {cY})", (cX + 10, cY + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            
            frame_to_save = cv2.cvtColor(frame_segmented, cv2.COLOR_RGB2BGR)
            
        out.write(frame_to_save)
    
    out.release()
    print(f"[Info] Video saved to {save_path}")


def select_frame_interactively(video_path, default_frame_idx=0, jump_seconds=0.2):
    """
    Select the frame interactively

    video_path: path to the video file
    default_frame_idx: default frame index
    jump_seconds: jump seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30 
    jump_frames = int(fps * jump_seconds)
    current_frame_idx = default_frame_idx

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Cannot read frame. Current frame index: {current_frame_idx}")
            break

        time_sec = current_frame_idx / fps

        cv2.putText(frame, f"Frame: {current_frame_idx}  Time: {time_sec:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Left (a): -{jump_seconds}s, Right (d): +{jump_seconds}s, s: select, q: cancel", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(f"Frame selection (a/d: {jump_seconds}s move, s: select, q: cancel)", frame)
        
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):  # select
            break
        elif key == ord('q'):  # cancel
            current_frame_idx = default_frame_idx
            break
        elif key == ord('a'):  # left move
            current_frame_idx = max(0, current_frame_idx - jump_frames)
        elif key == ord('d'):  # right move
            current_frame_idx = min(total_frames - 1, current_frame_idx + jump_frames)
        else:
            print("Invalid key. Please use a, d, s or q key.")

    cap.release()
    cv2.destroyAllWindows()
    return current_frame_idx