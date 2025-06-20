import cv2
import numpy as np
import math

class IrisProcessor:
    def __init__(self):
        # Parameter untuk deteksi iris
        self.pupil_min_radius = 10
        self.pupil_max_radius = 50
        self.iris_min_radius = 50
        self.iris_max_radius = 150
        
    def detect_pupil(self, img):
        """Mendeteksi pupil menggunakan Hough Circle Transform"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Deteksi lingkaran pupil
        circles = cv2.HoughCircles(
            gray, 
            cv2.HOUGH_GRADIENT, 
            1, 
            20,
            param1=50,
            param2=30,
            minRadius=self.pupil_min_radius,
            maxRadius=self.pupil_max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Ambil lingkaran pertama
            pupil = circles[0][0]
            return (pupil[0], pupil[1], pupil[2])
        return None

    def detect_iris(self, img, pupil_position):
        """Mendeteksi batas iris menggunakan integro-differential operator"""
        if pupil_position is None:
            return None
            
        x, y, _ = pupil_position
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        
        # Gunakan ROI sekitar pupil untuk efisiensi
        roi_size = self.iris_max_radius * 2
        x1 = max(0, x - roi_size)
        y1 = max(0, y - roi_size)
        x2 = min(gray.shape[1], x + roi_size)
        y2 = min(gray.shape[0], y + roi_size)
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Deteksi lingkaran iris
        circles = cv2.HoughCircles(
            roi, 
            cv2.HOUGH_GRADIENT, 
            1, 
            20,
            param1=50,
            param2=30,
            minRadius=self.iris_min_radius,
            maxRadius=self.iris_max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Pilih lingkaran terdekat dengan pusat pupil
            best_circle = None
            min_dist = float('inf')
            
            for circle in circles[0, :]:
                cx, cy, cr = circle
                dist = math.sqrt((cx - (x - x1))**2 + (cy - (y - y1))**2)
                if dist < min_dist:
                    min_dist = dist
                    best_circle = (cx + x1, cy + y1, cr)
            
            return best_circle
        return None

    def normalize_iris(self, img, pupil_circle, iris_circle):
        """Normalisasi iris menggunakan rubber sheet model (Daugman)"""
        if pupil_circle is None or iris_circle is None:
            return None
            
        px, py, pr = pupil_circle
        ix, iy, ir = iris_circle
        
        # Buat gambar hasil normalisasi
        height, width = 64, 256  # Dimensi hasil normalisasi
        normalized = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Konversi ke koordinat polar
        for y in range(height):
            for x in range(width):
                # r: 0 (pupil) hingga 1 (iris)
                r = y / height
                theta = 2 * np.pi * x / width
                
                # Hitung koordinat di gambar asli
                radius = pr + r * (ir - pr)
                src_x = int(px + radius * math.cos(theta))
                src_y = int(py + radius * math.sin(theta))
                
                # Ambil nilai pixel jika dalam batas gambar
                if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                    normalized[y, x] = img[src_y, src_x]
        
        return normalized

    def extract_features(self, normalized_iris):
        """Ekstraksi fitur menggunakan filter Gabor dan pengkodean biner"""
        if normalized_iris is None:
            return None, None
            
        gray = cv2.cvtColor(normalized_iris, cv2.COLOR_BGR2GRAY)
        
        # Aplikasikan filter Gabor
        kernel = cv2.getGaborKernel(
            ksize=(35, 35),
            sigma=4.0,
            theta=np.pi/4,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        
        # Binerisasi menggunakan Otsu's method
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Konversi ke template biner
        template = np.where(binary > 128, 1, 0).astype(np.uint8)
        
        # Buat mask (area yang valid)
        mask = np.ones_like(template, dtype=np.uint8)
        
        # Konversi ke byte array
        template_bytes = template.tobytes()
        mask_bytes = mask.tobytes()
        
        return template_bytes, mask_bytes

    def calculate_hamming_distance(self, template1, mask1, template2, mask2):
        """Menghitung jarak Hamming antara dua template iris"""
        if not template1 or not template2:
            return 1.0
            
        # Konversi byte array ke numpy array
        arr1 = np.frombuffer(template1, dtype=np.uint8)
        arr2 = np.frombuffer(template2, dtype=np.uint8)
        mask_arr1 = np.frombuffer(mask1, dtype=np.uint8)
        mask_arr2 = np.frombuffer(mask2, dtype=np.uint8)
        
        # Hitung mask gabungan
        combined_mask = mask_arr1 & mask_arr2
        valid_bits = np.count_nonzero(combined_mask)
        
        if valid_bits == 0:
            return 1.0
            
        # Hitung jarak Hamming
        xor_result = np.bitwise_xor(arr1, arr2)
        xor_result = np.bitwise_and(xor_result, combined_mask)
        hamming_dist = np.count_nonzero(xor_result) / valid_bits
        
        return hamming_dist