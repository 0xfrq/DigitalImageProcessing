import os
import numpy as np
from PIL import Image

from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor
from src.database import FaceDatabase

def get_cv2_cascade_path():
    cv2_data_dir = r"C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\cv2\data"
    if not os.path.exists(cv2_data_dir):
        import site
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            potential_path = os.path.join(sp, 'cv2', 'data')
            if os.path.exists(potential_path):
                cv2_data_dir = potential_path
                break
    return cv2_data_dir

def main():
    print("Memuat model Deteksi Mata (Manual Cascade)...")
    data_dir = get_cv2_cascade_path()
    eye_cascade = ManualCascadeClassifier(os.path.join(data_dir, 'haarcascade_eye.xml'))

    # Kita ubah arsitekturnya: semua distandarkan ke 100x100 di ekstraktor
    extractor = FaceFeatureExtractor(target_size=(100, 100))
    
    # HAPUS database lama agar tidak terjadi penumpukan dengan data format baru
    if os.path.exists("database_fitur.json"):
        os.remove("database_fitur.json")
    db = FaceDatabase("database_fitur.json") 
    
    input_folder = "hasil_deteksi"
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} tidak ditemukan!")
        return

    print("\nMemulai ekstraksi fitur biologis HANYA pada area wajah...")
    for image_name in os.listdir(input_folder):
        if not image_name.endswith('face detected.png'):
            continue
            
        nama_orang = image_name.replace('_face detected.png', '').split('(')[0].strip().lower()
        image_path = os.path.join(input_folder, image_name)
        
        try:
            pil_img = Image.open(image_path).convert('RGB')
            face_crop_rgb = np.array(pil_img)
            h_orig, w_orig, _ = face_crop_rgb.shape
        except Exception as e:
            print(f"Gagal memuat {image_name}: {e}")
            continue

        # 1. Tahap Deteksi Mata di Crop Asli
        gray = rgb_to_gray_manual(face_crop_rgb)
        gray = equalize_hist(gray)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))
        filtered_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey < h_orig * 0.6 and ew > 10 and eh > 10]
        filtered_eyes = sorted(filtered_eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

        # Konversi koordinat mata ke skala 100x100
        eyes_100 = []
        for (ex, ey, ew, eh) in filtered_eyes:
            nx = int(ex * 100 / w_orig)
            ny = int(ey * 100 / h_orig)
            nw = int(ew * 100 / w_orig)
            nh = int(eh * 100 / h_orig)
            eyes_100.append((nx, ny, nw, nh))

        # 2. Resize wajah ke 100x100
        face_100 = np.array(pil_img.resize((100, 100)))

        # 3. Ekstraksi Fitur Aktual pada CROP Wajah 100x100 (Preprocessing dilakukan di dalam)
        ekstraksi_result = extractor.extract_features(face_100, eyes_100)
        fitur_wajah = ekstraksi_result["features"]
        masks = ekstraksi_result["masks"]

        # 4. Simpan Mask untuk Visualisasi
        skin_path = os.path.join(input_folder, image_name.replace('face detected.png', 'skin_crop.png'))
        edge_path = os.path.join(input_folder, image_name.replace('face detected.png', 'edge_crop.png'))
        contour_path = os.path.join(input_folder, image_name.replace('face detected.png', 'contour_crop.png'))
        
        Image.fromarray(masks["skin"], mode='L').save(skin_path)
        Image.fromarray(masks["edge"], mode='L').save(edge_path)
        Image.fromarray(masks["contour"], mode='L').save(contour_path)

        # 5. Profiling (Database JSON)
        db.add_profile(nama_orang, fitur_wajah)
        print(f"[{image_name}] Merekam profil '{nama_orang}' dengan {len(eyes_100)} mata")

    print("\nProses selesai, menyusun database...")
    db.save()

if __name__ == "__main__":
    main()
