import cv2, numpy as np, sys, os
sys.path.append(os.getcwd())
from PIL import Image
from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import equalize_hist, rgb_to_gray_manual

cascade = ManualCascadeClassifier(r'C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\cv2\data\haarcascade_frontalface_default.xml')

gray_w = equalize_hist(rgb_to_gray_manual(np.array(Image.open('webcam/colorful_2 (2).png').convert('RGB').resize((320, 240)))))
gray_d = equalize_hist(rgb_to_gray_manual(np.array(Image.open('webcam/colorful_2.png').convert('RGB').resize((320, 240)))))

params = [(1.1, 2, 30), (1.1, 3, 30), (1.05, 2, 30), (1.05, 1, 60), (1.15, 1, 30)]

for s, n, m in params:
    w = len(cascade.detectMultiScale(gray_w, scaleFactor=s, minNeighbors=n, minSize=(m, m)))
    d = len(cascade.detectMultiScale(gray_d, scaleFactor=s, minNeighbors=n, minSize=(m, m)))
    print(f's={s}, n={n}, min={m} -> Woman:{w}, Doc:{d}')
