"""
visualisasi_pencocokan.py
-------------------------
File khusus Debugging & Visualisasi Komparatif (Tahap 6 & 7).
Menampilkan perbandingan visual antara gambar input dengan gambar di database.

Cara Pakai:
    python visualisasi_pencocokan.py <path_gambar_input> [--threshold 12.0] [--metode euclidean]

Contoh:
    python visualisasi_pencocokan.py webcam/face.jpg
"""

import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from PIL import Image

from src.manual_cascade import ManualCascadeClassifier
from src.image_utils import rgb_to_gray_manual, equalize_hist
from src.feature_extractor import FaceFeatureExtractor
from src.pencocokan import FaceMatcher

# ============================================================
# HELPER: Cari path XML Haar Cascade
# ============================================================
def find_cascade_xml(filename):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if os.path.exists(local):
        return local
    try:
        import site
        all_site = site.getsitepackages()
        try: all_site += [site.getusersitepackages()]
        except Exception: pass
        for sp in all_site:
            p = os.path.join(sp, 'cv2', 'data', filename)
            if os.path.exists(p): return p
    except Exception: pass
    for p in [f"/usr/share/opencv4/haarcascades/{filename}",
              f"/usr/local/share/opencv4/haarcascades/{filename}"]:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"'{filename}' tidak ditemukan.")

# ============================================================
# HELPER: Menggambar Titik Struktur Wajah (100x100)
# ============================================================
def gambar_struktur_wajah(ax, img_100, fitur, title_suffix=""):
    ax.imshow(img_100)
    ax.axis('off')
    
    eye_coords = fitur.get("eye_coords", [])
    nose_coords = fitur.get("nose_coords", [])
    mouth_coords = fitur.get("mouth_coords", [])
    eye_dist = fitur.get("eye_distance", 0)
    
    pe = [path_effects.withStroke(linewidth=2, foreground='black')]
    
    # Render Mata (Merah)
    if len(eye_coords) == 2:
        c1, c2 = eye_coords
        ax.plot(c1[0], c1[1], 'ro', markersize=6, markeredgecolor='white')
        ax.plot(c2[0], c2[1], 'ro', markersize=6, markeredgecolor='white')
        # Garis penghubung jarak mata
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--', linewidth=1.5)
        ax.text((c1[0]+c2[0])/2, ((c1[1]+c2[1])/2)-4, f"D:{eye_dist:.1f}", 
                color='red', ha='center', fontsize=9, path_effects=pe, fontweight='bold')
        
    # Render Hidung (Biru/Sian)
    if len(nose_coords) == 1:
        nx, ny = nose_coords[0]
        ax.plot(nx, ny, 'bo', markersize=6, markeredgecolor='white')
        
    # Render Mulut (Hijau)
    if len(mouth_coords) == 1:
        mx, my = mouth_coords[0]
        ax.plot(mx, my, 'go', markersize=6, markeredgecolor='white')

# ============================================================
# MAIN FUNCTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Visual Debugger Pencocokan Wajah")
    parser.add_argument("gambar", help="Path ke file gambar input (webcam/test)")
    parser.add_argument("--threshold", "-t", type=float, default=None)
    parser.add_argument("--metode", "-m", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    parser.add_argument("--database", "-d", default="database_fitur.json")
    args = parser.parse_args()

    if not os.path.exists(args.gambar):
        print(f"[ERROR] Gambar input '{args.gambar}' tidak ditemukan!")
        sys.exit(1)

    # 1. Inisialisasi Pipa Deteksi & Pencocokan
    face_cascade = ManualCascadeClassifier(find_cascade_xml('haarcascade_frontalface_default.xml'))
    eye_cascade  = ManualCascadeClassifier(find_cascade_xml('haarcascade_eye.xml'))
    extractor = FaceFeatureExtractor(target_size=(100, 100))
    matcher = FaceMatcher(db_path=args.database)

    # 2. Proses Gambar Input
    try:
        pil_img = Image.open(args.gambar).convert('RGB')
        img_orig = np.array(pil_img)
    except Exception as e:
        print(f"[ERROR] Gagal membaca gambar input: {e}")
        sys.exit(1)

    h_frame, w_frame, _ = img_orig.shape
    gray = rgb_to_gray_manual(img_orig)
    gray_eq = equalize_hist(gray)

    faces_raw = face_cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    faces = [(x, y, w, h) for (x, y, w, h) in faces_raw if w >= 60 and h >= 60]

    # Crop Wajah Input
    if len(faces) == 0:
        face_crop = img_orig
        print("[PERINGATAN] Wajah input tidak terdeteksi, menggunakan seluruh area gambar.")
    else:
        x, y, w, h = faces[0]
        h_ext = min(int(h * 1.25), h_frame - y)
        face_crop = img_orig[y:y + h_ext, x:x + w]

    h_crop, w_crop, _ = face_crop.shape
    gray_crop = rgb_to_gray_manual(face_crop)
    gray_crop_eq = equalize_hist(gray_crop)

    # Deteksi Mata Input
    eyes_raw = eye_cascade.detectMultiScale(gray_crop_eq, scaleFactor=1.05, minNeighbors=5, minSize=(15, 15))
    filtered_eyes = sorted([e for e in eyes_raw if e[1] < h_crop * 0.6 and e[2] > 10], key=lambda e: e[2]*e[3], reverse=True)[:2]
    eyes_100 = [(int(ex*100/w_crop), int(ey*100/h_crop), int(ew*100/w_crop), int(eh*100/h_crop)) for (ex, ey, ew, eh) in filtered_eyes]

    # Ekstraksi Fitur Gambar Input
    face_input_100 = np.array(Image.fromarray(face_crop).resize((100, 100)))
    hasil_ekstraksi_input = extractor.extract_features(face_input_100, eyes_100)
    fitur_input = hasil_ekstraksi_input["features"]

    # 3. Lakukan Pencocokan ke Database
    hasil_match = matcher.match(fitur_input, threshold=args.threshold, metode=args.metode)
    kandidat_nama = hasil_match["nama_kandidat"]
    
    # 4. Ambil Data Gambar & Fitur dari Database untuk Pembanding
    with open(args.database, 'r') as f:
        db_raw = json.load(f)
    fitur_database = db_raw[kandidat_nama][-1] # Ambil baris data fitur terakhir milik orang tersebut

    # Cari file gambar hasil deteksi milik profil di database untuk ditampilkan
    search_pattern = os.path.join("hasil_deteksi", f"{kandidat_nama}*_face detected.png")
    matched_db_images = glob.glob(search_pattern)
    
    if matched_db_images:
        img_db_orig = cv2.imread(matched_db_images[0])
        img_db_orig = cv2.cvtColor(img_db_orig, cv2.COLOR_BGR2RGB)
        img_db_100 = cv2.resize(img_db_orig, (100, 100))
    else:
        # Fallback jika gambar hasil_deteksi terhapus, buat gambar kosong/placeholder
        img_db_100 = np.zeros((100, 100, 3), dtype=np.uint8) + 100 
        print(f"[PERINGATAN] File cetakan wajah hasil_deteksi untuk '{kandidat_nama}' tidak ditemukan.")

    # ============================================================
    # 5. PEMBUATAN GRAFIK VISUALISASI (MATPLOTLIB)
    # ============================================================
    fig = plt.figure(figsize=(15, 9))
    fig.canvas.manager.set_window_title(f"DEBUGGER: Komparasi Vektor Fitur Wajah ({args.metode.upper()})")

    # Matriks Grid 2 baris x 3 kolom
    # Baris 1: 3 Gambar (Input Asli, Input Jarak Fitur, DB Jarak Fitur)
    ax_input_asli = plt.subplot(2, 3, 1)
    ax_input_asli.set_title("1. Gambar Input Asli", fontsize=12, pad=10, fontweight='bold')
    ax_input_asli.imshow(img_orig)
    ax_input_asli.axis('off')
    if len(faces) > 0:
        # Beri kotak penanda wajah di gambar asli jika terdeteksi
        rect = patches.Rectangle((x, y), w, h_ext, linewidth=2, edgecolor='g', facecolor='none')
        ax_input_asli.add_patch(rect)

    ax_input_crop = plt.subplot(2, 3, 2)
    ax_input_crop.set_title("2. Fitur Lokal Input (100x100)", fontsize=12, pad=10, fontweight='bold')
    gambar_struktur_wajah(ax_input_crop, face_input_100, fitur_input)

    ax_db_crop = plt.subplot(2, 3, 3)
    ax_db_crop.set_title(f"3. Fitur Database: '{kandidat_nama.upper()}'", fontsize=12, pad=10, fontweight='bold')
    gambar_struktur_wajah(ax_db_crop, img_db_100, fitur_database)

    # Baris 2, Kolom 1 & 2 gabung: Bar Chart Komparasi Fitur-Fitur Utama
    ax_chart = plt.subplot(2, 3, (4, 5))
    ax_chart.set_title("Perbandingan Karakteristik Biologis & Tekstur", fontsize=12, pad=10, fontweight='bold')
    
    kunci_komparasi = ["eye_distance", "skin_ratio", "edge_density", "contour_density", "symmetry_error"]
    labels_piagam = ["Jarak Mata", "Rasio Kulit", "Kerapatan Tepi", "Kontur Kulit", "Error Simetri"]
    
    nilai_input = [float(fitur_input.get(k, 0)) for k in kunci_komparasi]
    nilai_db = [float(fitur_database.get(k, 0)) for k in kunci_komparasi]
    
    y_pos = np.arange(len(labels_piagam))
    width = 0.35
    
    ax_chart.barh(y_pos - width/2, nilai_input, width, label='Gambar Input', color='#3498db')
    ax_chart.barh(y_pos + width/2, nilai_db, width, label=f'DB ({kandidat_nama})', color='#e67e22')
    
    ax_chart.set_yticks(y_pos)
    ax_chart.set_yticklabels(labels_piagam, fontsize=11)
    ax_chart.invert_yaxis()
    ax_chart.legend()
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)

    # Baris 2, Kolom 3: Hasil Keputusan Akhir Keaslian / Kesamaan
    ax_status = plt.subplot(2, 3, 6)
    ax_status.axis('off')
    
    status_text = "DIKENALI" if hasil_match["dikenali"] else "TIDAK DIKENALI"
    status_color = "green" if hasil_match["dikenali"] else "red"
    status_icon = "✅" if hasil_match["dikenali"] else "❌"
    
    # Render kotak info keputusan
    box_style = dict(boxstyle='round,pad=1', facecolor='#f5f5f5', edgecolor='gray', alpha=0.5)
    ax_status.text(0.5, 0.5, 
                   f"STATUS VERIFIKASI:\n"
                   f"{status_icon} {status_text}\n\n"
                   f"Identitas Terdekat:\n"
                   f"👉 {kandidat_nama.upper()}\n\n"
                   f"Skor Jarak ({args.metode}):\n"
                   f"📊 {hasil_match['jarak']:.4f}\n"
                   f"Batas Threshold:\n"
                   f"🏁 {hasil_match['threshold']}",
                   fontsize=12, fontweight='bold', color='black',
                   ha='center', va='center', bbox=box_style)
    
    # Tambahkan warna khusus pada status teks di dalam matplotlib
    fig.text(0.83, 0.32, f"       ", color=status_color, weight='bold', fontsize=14)

    plt.tight_layout()
    
    # Save hasil debug ke file lokal
    output_debug_path = "debug_hasil_pencocokan.png"
    plt.savefig(output_debug_path, dpi=150)
    print(f"\n[SUKSES] Visualisasi debug berhasil dibuat dan disimpan ke: {output_debug_path}")
    
    plt.show()

if __name__ == "__main__":
    main()