import cv2
import os
import random
import shutil
from sklearn.model_selection import train_test_split

# ------------------------
# SETTINGS
# ------------------------
video_dataset_dir = "D:/DeepFake/v/DFD"  # your folder with videos (e.g., fake and real subfolders inside)
output_dir = "D:/DeepFake/pythonProject1/Frames/FF/FF 600"  # where extracted frames will be stored
frame_interval = 30  # extract 1 frame every 30 frames (~1 frame per second if 30fps)
img_size = (600, 600)  # resize extracted frames (match your pretrained models)
extensions = (".mp4", ".avi", ".mov")  # supported video formats

# ------------------------
# CREATE OUTPUT FOLDERS
# ------------------------
splits = ["train", "validation", "test"]
categories = os.listdir(video_dataset_dir)

for split in splits:
    for cat in categories:
        os.makedirs(os.path.join(output_dir, split, cat), exist_ok=True)


# ------------------------
# EXTRACT FRAMES
# ------------------------
def extract_frames_from_video(video_path, save_dir, frame_interval=30, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_images = []
    success, frame = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, img_size)
            filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_images.append(filepath)
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    return saved_images


all_images = {cat: [] for cat in categories}

for cat in categories:
    video_files = [f for f in os.listdir(os.path.join(video_dataset_dir, cat)) if f.endswith(extensions)]
    for video_file in video_files:
        video_path = os.path.join(video_dataset_dir, cat, video_file)
        save_dir = os.path.join(output_dir, "all", cat)
        os.makedirs(save_dir, exist_ok=True)
        extracted = extract_frames_from_video(video_path, save_dir, frame_interval, img_size)
        all_images[cat].extend(extracted)

# ------------------------
# SPLIT INTO TRAIN/VAL/TEST
# ------------------------
for cat, images in all_images.items():
    train_val, test = train_test_split(images, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # 0.1765 * 0.85 ≈ 0.15

    split_map = {
        "train": train,
        "validation": val,
        "test": test
    }

    for split, files in split_map.items():
        for file in files:
            shutil.move(file, os.path.join(output_dir, split, cat, os.path.basename(file)))

# ------------------------
# CLEAN UP TEMP "all" FOLDER
# ------------------------
shutil.rmtree(os.path.join(output_dir, "all"))

print("✅ Frame extraction and dataset split completed successfully!")
