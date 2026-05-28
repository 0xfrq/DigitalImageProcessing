import json
import os

class FaceDatabase:
    def __init__(self, db_path="database_fitur.json"):
        self.db_path = db_path
        self.data = {}
        self.load()

    def load(self):
        """Membaca data profil orang dari file database."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                try:
                    self.data = json.load(f)
                except json.JSONDecodeError:
                    self.data = {}
        else:
            self.data = {}

    def add_profile(self, name, features):
        """
        Menambahkan profil fitur wajah baru untuk orang tertentu.
        Satu orang bisa memiliki banyak entri fitur (jika ada lebih dari 1 foto).
        """
        if name not in self.data:
            self.data[name] = []
            
        self.data[name].append(features)

    def save(self):
        """Menyimpan data kembali ke file secara terstruktur."""
        with open(self.db_path, 'w') as f:
            json.dump(self.data, f, indent=4)
        print(f"-> Database berhasil disimpan di {self.db_path}")

    def get_all_profiles(self):
        """Mengembalikan seluruh profil yang terdaftar."""
        return self.data
