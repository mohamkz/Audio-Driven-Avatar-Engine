import cv2
import numpy as np
import librosa
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import Wav2Vec2Processor, Wav2Vec2Model

LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

def get_mediapipe_landmarker(model_path="face_landmarker.task"):
    import urllib.request
    import os

    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def extract_lip_coordinates(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    lip_coords = []
    last_known = [[0.5, 0.5]] * len(LIP_INDICES)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((frame_idx / fps) * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if results.face_landmarks:
            face = results.face_landmarks[0]
            current = [[face[idx].x, face[idx].y] for idx in LIP_INDICES]
            lip_coords.append(current)
            last_known = current
        else:
            lip_coords.append(last_known)
        frame_idx += 1
    cap.release()
    return np.array(lip_coords)

def extract_audio_features(video_path, device="cpu"):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    speech, _ = librosa.load(video_path, sr=16000)
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    with torch.no_grad():
        return model(inputs).last_hidden_state.squeeze().cpu().numpy()