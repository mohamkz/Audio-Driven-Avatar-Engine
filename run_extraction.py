import os
import numpy as np
from src.extraction import (
    get_mediapipe_landmarker, 
    extract_lip_coordinates, 
    extract_audio_features
)
from dotenv import load_dotenv

load_dotenv()

os.makedirs("data", exist_ok=True)

video_input = "data/full_podcast.mp4"
audio_output = "data/X_audio_data.npy"
lip_output = "data/Y_lip_data.npy"

def run():
    if not os.path.exists(video_input):
        print(f"Error: Place your video at {video_input} before running.")
        return

    print("Initializing MediaPipe...")
    landmarker = get_mediapipe_landmarker()

    print("Extracting Lip Coordinates...")
    y_data = extract_lip_coordinates(video_input, landmarker)
    np.save(lip_output, y_data)
    print(f"Saved visuals: {y_data.shape}")

    print("Extracting Audio Features (CUDA)...")
    x_data = extract_audio_features(video_input, device="cuda")
    np.save(audio_output, x_data)
    print(f"Saved audio: {x_data.shape}")

    print("\n Preprocessing complete.")

if __name__ == "__main__":
    run()