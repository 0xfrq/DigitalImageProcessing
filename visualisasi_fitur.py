import json
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches

database_file = "database_fitur.json"
image_folder = "hasil_deteksi"
original_folder = "webcam"

if not os.path.exists(database_file):
    print("Database JSON tidak ditemukan!")
    exit()

with open(database_file, 'r') as f:
    data = json.load(f)

if not data:
    print("Database kosong!")
    exit()

nama_profil = list(data.keys())[0]
fitur = data[nama_profil][-1] # Ambil data TERBARU

# Ekstrak semua fitur yang baru
eye_dist = fitur.get("eye_distance", 0)
eye_coords = fitur.get("eye_coords", [])
nose_coords = fitur.get("nose_coords", [])
mouth_coords = fitur.get("mouth_coords", [])

skin_ratio = fitur.get("skin_ratio", 0)
edge_density = fitur.get("edge_density", 0)
contour_density = fitur.get("contour_density", 0)
eye_region_edge = fitur.get("eye_region_edge", 0)
eye_non_skin = fitur.get("eye_non_skin", 0)
nose_region_edge = fitur.get("nose_region_edge", 0)
mouth_non_skin = fitur.get("mouth_non_skin", 0)

# Cari gambar crop
search_pattern = os.path.join(image_folder, f"{nama_profil}*_face detected.png")
matched_images = glob.glob(search_pattern)

if not matched_images:
    print(f"Gambar crop untuk profil '{nama_profil}' tidak ditemukan!")
    exit()

img_path = matched_images[0]
img_crop_orig = cv2.imread(img_path)
img_crop_orig = cv2.cvtColor(img_crop_orig, cv2.COLOR_BGR2RGB)

# KARENA DI EKSTRAKTOR KITA RESIZE KE 100x100
# Semua koordinat dari DB sekarang merujuk ke 100x100!
img_100 = cv2.resize(img_crop_orig, (100, 100))

# Cari gambar asli
base_name = os.path.basename(img_path).replace("_face detected.png", "")
orig_pattern = os.path.join(original_folder, f"{base_name}*")
matched_orig = glob.glob(orig_pattern)

img_orig = None
if matched_orig:
    for f_path in matched_orig:
        if os.path.isfile(f_path):
            img_orig = cv2.imread(f_path)
            if img_orig is not None:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                break

def load_mask(suffix):
    p = os.path.join(image_folder, f"{nama_profil}*_{suffix}.png")
    m = glob.glob(p)
    if m:
        return cv2.imread(m[0], cv2.IMREAD_GRAYSCALE)
    return None

skin_img = load_mask("skin_crop")
edge_img = load_mask("edge_crop")
contour_img = load_mask("contour_crop")

fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title('Visualisasi Struktur Wajah (DIP-based)')

# 1. GAMBAR ASLI
ax1 = plt.subplot(2, 3, 1)
ax1.set_title("1. Gambar Asli", fontsize=14, pad=15)
if img_orig is not None:
    ax1.imshow(img_orig)
ax1.axis('off')

# 2. CROP + TITIK KOORDINAT
ax2 = plt.subplot(2, 3, 2)
ax2.set_title("2. Deteksi Struktur Lokal (100x100)", fontsize=14, pad=15)
ax2.imshow(img_100)
ax2.axis('off')

pe = [path_effects.withStroke(linewidth=2, foreground='black')]

# Gambar Mata (Merah)
if len(eye_coords) == 2:
    c1, c2 = eye_coords
    ax2.plot(c1[0], c1[1], 'ro', markersize=6, markeredgecolor='white')
    ax2.plot(c2[0], c2[1], 'ro', markersize=6, markeredgecolor='white')
    # Area Mata
    rect = patches.Rectangle((0, 0), 100, 50, linewidth=1, edgecolor='red', facecolor='none', linestyle=':')
    ax2.add_patch(rect)
    ax2.text(50, 15, "Area Mata", color='red', ha='center', fontsize=9, path_effects=pe)

# Gambar Hidung (Biru)
if len(nose_coords) == 1:
    nx, ny = nose_coords[0]
    ax2.plot(nx, ny, 'bo', markersize=6, markeredgecolor='white')
    # Area Hidung
    rect = patches.Rectangle((35, 40), 30, 35, linewidth=1, edgecolor='blue', facecolor='none', linestyle=':')
    ax2.add_patch(rect)
    ax2.text(nx, ny-5, "Hidung", color='cyan', ha='center', fontsize=9, path_effects=pe)

# Gambar Mulut (Hijau)
if len(mouth_coords) == 1:
    mx, my = mouth_coords[0]
    ax2.plot(mx, my, 'go', markersize=6, markeredgecolor='white')
    # Area Mulut
    rect = patches.Rectangle((25, 70), 50, 25, linewidth=1, edgecolor='lime', facecolor='none', linestyle=':')
    ax2.add_patch(rect)
    ax2.text(mx, my-5, "Mulut", color='lime', ha='center', fontsize=9, path_effects=pe)

# 3. SKIN MASK
ax3 = plt.subplot(2, 3, 3)
ax3.set_title(f"3. Segmentasi Kulit", fontsize=14, pad=15)
if skin_img is not None:
    ax3.imshow(skin_img, cmap='gray')
ax3.axis('off')

# 4. EDGE MASK
ax4 = plt.subplot(2, 3, 4)
ax4.set_title(f"4. Deteksi Tepi (Sobel)", fontsize=14, pad=15)
if edge_img is not None:
    ax4.imshow(edge_img, cmap='gray')
ax4.axis('off')

# 5. KONTUR WAJAH
ax5 = plt.subplot(2, 3, 5)
ax5.set_title(f"5. Kontur Area Kulit", fontsize=14, pad=15)
if contour_img is not None:
    ax5.imshow(contour_img, cmap='gray')
ax5.axis('off')

# 6. RINGKASAN DATA
ax6 = plt.subplot(2, 3, 6)
ax6.set_title("6. Karakteristik Biologis Database", fontsize=14, pad=15)

labels = ['Global Skin', 'Global Edge', 'Face Contour', 'Eye Edge', 'Eye Non-Skin', 'Nose Edge', 'Mouth Non-Skin']
values = [skin_ratio, edge_density, contour_density, eye_region_edge, eye_non_skin, nose_region_edge, mouth_non_skin]

y_pos = np.arange(len(labels))
ax6.barh(y_pos, values, color=['#e67e22', '#3498db', '#f1c40f', '#e74c3c', '#9b59b6', '#2ecc71', '#1abc9c'])
ax6.set_yticks(y_pos)
ax6.set_yticklabels(labels)
ax6.invert_yaxis()  # labels read top-to-bottom
ax6.set_xlim(0, 1)

for i, v in enumerate(values):
    ax6.text(v + 0.02, i + 0.1, f"{v*100:.1f}%", color='black', fontweight='bold', fontsize=10)

ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.set_xticks([])

plt.tight_layout()
output_img = "visualisasi_fitur_aktual.png"
plt.savefig(output_img, dpi=150)
print(f"Visualisasi berhasil di-update dan disimpan ke {output_img}!")
plt.show()
