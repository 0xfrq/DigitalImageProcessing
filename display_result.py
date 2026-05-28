import os
import glob
import random
from matplotlib import pyplot as plt
from src.image_utils import read_pgm, equalize_hist, gaussian_blur, laplacian, gamma_correction

base_dir = os.path.dirname(__file__)
folder_name = 'dataset'
path_to_dataset = os.path.join(base_dir, folder_name)

def load_and_process_dataset(path_folder):
    file_pattern = os.path.join(path_folder, "*.pgm")
    file_list = glob.glob(file_pattern)
    file_list.sort()

    if not file_list:
        print(f"File not found in: {path_folder}")
        return

    print(f"Succesfull find {len(file_list)} file .pgm!")

    num_samples = min(5, len(file_list))
    samples = random.sample(file_list, num_samples)

    num_cols = 4
    plt.figure(figsize=(18, 10))

    for i, file_path in enumerate(samples):
        try:
            img = read_pgm(file_path)

            img_equ = equalize_hist(img)
            img_blur = gaussian_blur(img_equ)
            img_laplacian = laplacian(img_blur)
            img_gamma_bright = gamma_correction(img, gamma=0.45)

            display_list = [img, img_equ, img_blur, img_laplacian, img_gamma_bright]
            titles = ['original', 'equalized', 'gaussian blur', 'laplacian', 'gamma gelap']

            for j in range(num_cols + 1):
                plt.subplot(num_samples, num_cols + 1, i * (num_cols + 1) + j + 1)
                plt.imshow(display_list[j], cmap='gray', vmin=0, vmax=255)
                if i == 0:
                    plt.title(titles[j], fontsize=7)
                plt.axis('off')

        except Exception as err:
            print(f"Error while processing '{file_path}': {err}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_process_dataset(path_to_dataset)