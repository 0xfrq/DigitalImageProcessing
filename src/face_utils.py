import numpy as np

def gaussian_blur_manual(image):
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float32)
    kernel = kernel / 16.0
    height, width = image.shape
    output = np.zeros_like(image)
    padded = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    for y in range(height):
        for x in range(width):
            region = padded[y:y+3, x:x+3]
            value = np.sum(region * kernel)
            output[y, x] = value
    return output

def sobel_edge_detection(image):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])
    height, width = image.shape
    output = np.zeros_like(image)
    padded = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    for y in range(height):
        for x in range(width):
            region = padded[y:y+3, x:x+3]
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            magnitude = min(255, magnitude)
            output[y, x] = magnitude
    return output.astype(np.uint8)

def skin_segmentation_manual(hsv):
    height, width, _ = hsv.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            h = hsv[y, x, 0]
            s = hsv[y, x, 1]
            v = hsv[y, x, 2]
            if 0 <= h <= 20 and 30 <= s <= 150 and 60 <= v <= 255:
                mask[y, x] = 255
    return mask
