import os
import glob
from src.image_utils import read_pgm, write_pgm, equalize_hist, gaussian_blur, laplacian, gamma_correction

base_dir = os.path.dirname(__file__)
folder_name = 'dataset'
path_to_dataset = os.path.join(base_dir, folder_name)
folder_output = 'dataset_hasil'
path_output = os.path.join(base_dir, folder_output)

if not os.path.exists(path_output):
    os.makedirs(path_output)
    print(f"Folder '{folder_output}' created successfully in {base_dir}")
else:
    print(f"Folder '{folder_output}' already available. New images will be added in it.")

def load_process_save_dataset(path_folder):
    file_pattern = os.path.join(path_folder, "*.pgm")
    file_list = glob.glob(file_pattern)
    file_list.sort()
 
    if not file_list:
        print(f"File not found in: {path_folder}")
        return
 
    print(f"Succesfull find {len(file_list)} file .pgm! Processing and saving...")
    processed_count = 0
 
    for i, file_path in enumerate(file_list, start=1):
        try:
            img = read_pgm(file_path)
  
            img_equ = equalize_hist(img)
            img_blur = gaussian_blur(img_equ)
            
            img_laplacian = laplacian(img_equ)
            img_gamma_bright = gamma_correction(img, gamma=0.45)
 
            base_filename        = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
 
            paths = {
                '_equ.pgm'          : img_equ,
                '_blur.pgm'         : img_blur,
                '_laplacian.pgm'    : img_laplacian,
                '_gamma_bright.pgm' : img_gamma_bright,
            }

            for suffix, result_img in paths.items():
                out_path = os.path.join(path_output, filename_without_ext + suffix)
                write_pgm(out_path, result_img)
 
            processed_count += 1
            if i % 10 == 0:
                print(f"{i} images have been successfully processed so far.")
 
        except Exception as err:
            print(f"Error while processing '{file_path}': {err}")
 
    print(f"All processed images have been saved in the folder: {folder_output}. Total saved: {processed_count}")

if __name__ == "__main__":
    load_process_save_dataset(path_to_dataset)