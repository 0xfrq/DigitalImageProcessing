import numpy as np

def read_pgm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().strip()
        if header != b'P5':
            raise ValueError(f"File {file_path} bukan format P5 PGM.")
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b'#'):
                f.seek(pos)
                break
        dim_line = f.readline().strip().split()
        width, height = int(dim_line[0]), int(dim_line[1])
        maxval = int(f.readline().strip())
        return np.fromfile(f, dtype=np.uint8).reshape((height, width))

def write_pgm(file_path, image):
    height, width = image.shape
    with open(file_path, 'wb') as f:
        f.write(b'P5\n')
        f.write(f"{width} {height}\n".encode('ascii'))
        f.write(b'255\n')
        f.write(image.astype(np.uint8).tobytes())

def equalize_hist(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = np.round((cdf_masked - cdf_masked.min()) * 255.0 / (cdf_masked.max() - cdf_masked.min()))
    cdf = np.ma.filled(cdf_masked, 0).astype(np.uint8)
    return cdf[img]

def convolve2d(image, kernel):
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(k_h):
        for j in range(k_w):
            output += padded[i : i + image.shape[0], j : j + image.shape[1]] * kernel[i, j]
    return output

def gaussian_blur(img):
    kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=np.float64) / 273.0
    blurred = convolve2d(img, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def laplacian(img):
    kernel = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float64)
    laplacian_img = convolve2d(img, kernel)
    return np.clip(np.absolute(laplacian_img), 0, 255).astype(np.uint8)

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return table[img]

def rgb_to_gray_manual(frame):
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    return gray.astype(np.uint8)
