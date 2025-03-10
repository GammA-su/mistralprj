import cv2
import numpy as np
import os

# Load all generated images
image_files = [f"consistent_waifu_look_{i}.png" for i in range(1, 16)]
images = []

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Warning: Could not load {img_path}, skipping...")
    else:
        images.append(img)

# Check if we have at least 2 images to create a video
if len(images) < 2:
    print("❌ Not enough valid images to create a video. Exiting...")
    exit()

# Get image dimensions
height, width, layers = images[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
fps = 30  # Frames per second
video = cv2.VideoWriter('waifu_transition.mp4', fourcc, fps, (width, height))

# Morphing effect: Interpolate between frames
num_blend_frames = 15  # Number of intermediate frames between two images
for i in range(len(images) - 1):
    for alpha in np.linspace(0, 1, num_blend_frames):
        blended = cv2.addWeighted(images[i], 1 - alpha, images[i + 1], alpha, 0)
        video.write(blended)

# Add last image as static frame
for _ in range(fps):  # Hold last frame for 1 second
    video.write(images[-1])

video.release()
print("✅ Video saved as 'waifu_transition.mp4'")
