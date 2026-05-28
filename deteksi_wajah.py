import cv2
import numpy as np
import os
import sys

from src.face_utils import gaussian_blur_manual, sobel_edge_detection, skin_segmentation_manual
from src.manual_cascade import ManualCascadeClassifier

# =========================
# LOAD CASCADE
# =========================
print("Loading manual Haar Cascade from XML... (this may take a moment)")
face_cascade = ManualCascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Note: For speed we still use cv2 for eyes, but face uses the scratch implementation!
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# =========================
# FOLDER INPUT / OUTPUT
# =========================
input_folder = "webcam"
output_skin = "hasil_skin"
output_edge = "hasil_edge"
output_detection = "hasil_deteksi"

os.makedirs(output_skin, exist_ok=True)
os.makedirs(output_edge, exist_ok=True)
os.makedirs(output_detection, exist_ok=True)

# =========================
# AMBIL SEMUA GAMBAR
# =========================
image_files = [f for f in os.listdir(input_folder) if f.endswith(".png") or f.endswith(".pgm")]

# =========================
# PROSES GAMBAR
# =========================
for image_name in image_files:
    print(f"Memproses: {image_name}")
    image_path = os.path.join(input_folder, image_name)
    frame = cv2.imread(image_path)
    
    if frame is None:
        continue

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    blurred = gaussian_blur_manual(gray)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = skin_segmentation_manual(hsv)
    edges = sobel_edge_detection(blurred)

    print(f"-> Detecting faces using from-scratch cascade on {image_name}...")
    # NOTE: It is very slow in pure python!
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=2, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # YOLO style bounding box
        color = (0, 255, 0) # Green color for bounding box
        thickness = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        label = "Face"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw filled rectangle for label background
        cv2.rectangle(frame, (x, max(0, y - label_height - baseline - 5)), (x + label_width, y), color, cv2.FILLED)
        # Put text
        cv2.putText(frame, label, (x, max(0, y - baseline - 2)), font, font_scale, (0, 0, 0), font_thickness)

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=9, minSize=(20, 20))
        filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < h * 0.5 and ew > 15 and eh > 15]
        
        filtered_eyes = sorted(filtered_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        filtered_eyes = sorted(filtered_eyes, key=lambda e: e[0])

        for (ex, ey, ew, eh) in filtered_eyes:
            center_eye = (ex + ew // 2, ey + eh // 2)
            radius_eye = int(ew * 0.45)
            cv2.circle(roi_color, center_eye, radius_eye, (255, 0, 0), 3)

    filename = os.path.splitext(image_name)[0]
    
    if len(faces) > 0:
        det_name = "face detected"
    else:
        det_name = "not detected"

    cv2.imwrite(os.path.join(output_skin, filename + "_skin.png"), skin_mask)
    cv2.imwrite(os.path.join(output_edge, filename + "_edge.png"), edges)
    cv2.imwrite(os.path.join(output_detection, f"{filename}_{det_name}.png"), frame)
    print(f"Selesai: {image_name}")

cv2.destroyAllWindows()
print("Semua gambar berhasil diproses")