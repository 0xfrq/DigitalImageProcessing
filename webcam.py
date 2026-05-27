import tkinter as tk
import numpy as np
import cv2
import os
from PIL import Image, ImageTk

def rgb_to_gray_manual(frame):
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
    return gray.astype(np.uint8)

def write_pgm(file_path, image):
    height, width = image.shape
    with open(file_path, 'wb') as f:
        f.write(b'P5\n')
        f.write(f"{width} {height}\n".encode('ascii'))
        f.write(b'255\n')
        f.write(image.astype(np.uint8).tobytes())

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Capture")
        self.root.geometry("650x750")

        self.save_dir = os.path.join(os.path.dirname(__file__), 'webcam')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.camera = None
        self.is_bursting = False
        self.current_frame = None

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(pady=10, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(root)
        self.control_frame.pack(pady=5)

        self.mode_var = tk.StringVar(value="color")
        tk.Radiobutton(self.control_frame, text="Colorful (.png)", variable=self.mode_var, value="color").grid(row=0, column=0, padx=10, pady=5)
        tk.Radiobutton(self.control_frame, text="Grayscale (.pgm)", variable=self.mode_var, value="gray").grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.control_frame, text="Interval Burst (Detik):").grid(row=1, column=0, pady=5, sticky="e")
        self.burst_slider = tk.Scale(self.control_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.burst_slider.set(0.5)
        self.burst_slider.grid(row=1, column=1, pady=5, sticky="w")

        tk.Label(self.control_frame, text="Pilih Kamera:").grid(row=2, column=0, pady=5, sticky="e")
        
        self.available_cams = self.scan_cameras()
        self.cam_var = tk.StringVar(value=self.available_cams[0])
        self.cam_menu = tk.OptionMenu(self.control_frame, self.cam_var, *self.available_cams, command=self.ganti_kamera)
        self.cam_menu.config(width=15)
        self.cam_menu.grid(row=2, column=1, pady=5, sticky="w")

        self.mirror_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.control_frame, text="Mirror View", variable=self.mirror_var).grid(row=3, column=0, columnspan=2, pady=5)

        self.btn_capture = tk.Button(root, text="Tahan untuk Burst / Klik untuk 1x", font=("Arial", 12, "bold"), bg="lightblue", width=35, height=2)
        self.btn_capture.pack(pady=10)

        self.status_label = tk.Label(root, text="Memulai kamera...", fg="blue")
        self.status_label.pack()

        self.btn_capture.bind("<ButtonPress-1>", self.start_capture)
        self.btn_capture.bind("<ButtonRelease-1>", self.stop_capture)

        self.nyalakan_kamera(self.cam_var.get())
        self.update_frame()

    def scan_cameras(self):
        active_ports = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    active_ports.append(f"Port {i}")
                cap.release()
        return active_ports if active_ports else ["Tidak ada kamera"]

    def nyalakan_kamera(self, cam_name):
        if cam_name == "Tidak ada kamera":
            self.status_label.config(text="Kamera tidak ditemukan!", fg="red")
            return
            
        cam_index = int(cam_name.split()[1])
        self.camera = cv2.VideoCapture(cam_index)
        if self.camera.isOpened():
            self.status_label.config(text=f"Kamera ({cam_name}) berhasil menyala!", fg="green")
        else:
            self.status_label.config(text=f"Gagal! Kamera ({cam_name}) tidak terdeteksi.", fg="red")
            self.camera.release()
            self.camera = None
            self.video_label.configure(image='')

    def ganti_kamera(self, pilihan_baru):
        self.status_label.config(text=f"Mencoba beralih ke {pilihan_baru}...", fg="blue")
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.nyalakan_kamera(pilihan_baru)

    def update_frame(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                frame = frame[:, :, ::-1]
                if self.mirror_var.get():
                    frame = frame[:, ::-1]
                self.current_frame = frame
                frame_display = frame[::2, ::2] 
                img = Image.fromarray(frame_display)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        self.root.after(30, self.update_frame)

    def get_next_counter(self, prefix, ext):
        files = os.listdir(self.save_dir)
        counter = 1
        while f"{prefix}_{counter}{ext}" in files:
            counter += 1
        return counter

    def start_capture(self, event):
        if self.camera is None or not self.camera.isOpened():
            return
        self.is_bursting = True
        self.btn_capture.config(bg="salmon", text="📸 Mengambil Gambar...")
        self.take_picture()

    def stop_capture(self, event):
        self.is_bursting = False
        self.btn_capture.config(bg="lightblue", text="Tahan untuk Burst / Klik untuk 1x")

    def take_picture(self):
        if not self.is_bursting or self.current_frame is None:
            return

        mode = self.mode_var.get()
        if mode == "gray":
            counter = self.get_next_counter("grayscale", ".pgm")
            filename = f"grayscale_{counter}.pgm"
            filepath = os.path.join(self.save_dir, filename)
            gray_img = rgb_to_gray_manual(self.current_frame)
            write_pgm(filepath, gray_img)
        else:
            counter = self.get_next_counter("colorful", ".png")
            filename = f"colorful_{counter}.png"
            filepath = os.path.join(self.save_dir, filename)
            Image.fromarray(self.current_frame).save(filepath)

        self.status_label.config(text=f"Tersimpan: {filename}", fg="green")
        interval_ms = int(self.burst_slider.get() * 1000)
        self.root.after(interval_ms, self.take_picture)

    def on_closing(self):
        if self.camera is not None:
            self.camera.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
