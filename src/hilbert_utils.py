import numpy as np

class HilbertCurveCompressor:
    def __init__(self, n):
        self.n = n
        self.size = 1 << n

    def _hilbert_curve(self, n):
        if n == 0:
            return np.array([[0, 0]])
        prev = self._hilbert_curve(n - 1)
        m = 1 << (n - 1)
        q1 = np.column_stack((prev[:, 1], prev[:, 0]))
        q2 = np.column_stack((prev[:, 0], prev[:, 1] + m))
        q3 = np.column_stack((prev[:, 0] + m, prev[:, 1] + m))
        q4 = np.column_stack((2 * m - 1 - prev[:, 1], m - 1 - prev[:, 0]))
        return np.vstack((q1, q2, q3, q4))

    def get_curve_indices(self):
        return self._hilbert_curve(self.n)

    def image_to_curve(self, gray_image):
        assert gray_image.shape == (self.size, self.size)
        curve = self.get_curve_indices()
        return np.array([gray_image[y, x] for y, x in curve], dtype=np.uint8)

    def curve_to_image(self, curve_1d):
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        curve = self.get_curve_indices()
        for idx, (y, x) in enumerate(curve):
            img[y, x] = curve_1d[idx]
        return img

def compress_grayscale(gray_img, hilbert_compressor):
    curve_data = hilbert_compressor.image_to_curve(gray_img)
    diff = np.diff(curve_data.astype(np.int16), prepend=curve_data[0])
    compressed = []
    run_val = diff[0]
    run_len = 1
    for val in diff[1:]:
        if val == run_val and run_len < 255:
            run_len += 1
        else:
            compressed.extend([run_val, run_len])
            run_val = val
            run_len = 1
    compressed.extend([run_val, run_len])
    return np.array(compressed, dtype=np.int16).tobytes()

def decompress_grayscale(compressed_bytes, hilbert_compressor, original_length):
    diff_rle = np.frombuffer(compressed_bytes, dtype=np.int16)
    diffs = []
    for i in range(0, len(diff_rle), 2):
        val = diff_rle[i]
        length = diff_rle[i + 1]
        diffs.extend([val] * length)
    diffs = np.array(diffs[:original_length], dtype=np.int16)
    curve_reconstructed = np.cumsum(diffs).astype(np.uint8)
    return hilbert_compressor.curve_to_image(curve_reconstructed)

def load_compressed_from_file(filepath, order=8):
    with open(filepath, 'rb') as f:
        compressed_bytes = f.read()
    hilbert_comp = HilbertCurveCompressor(order)
    original_length = hilbert_comp.size * hilbert_comp.size
    return decompress_grayscale(compressed_bytes, hilbert_comp, original_length)
