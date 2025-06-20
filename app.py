import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from Crypto.Cipher import AES
import pymysql
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import threading
import time
import atexit
import math

app = Flask(__name__)

# ===== CONFIGURATION =====
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_USER = os.environ.get('MYSQL_USER', 'iris_app')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'password_kuat123!')
MYSQL_DB = os.environ.get('MYSQL_DB', 'iris_security')
MYSQL_CHARSET = 'utf8mb4'

AES_KEY = os.environ.get('AES_KEY', 'my_super_secret_key_32bytes').ljust(32)[:32].encode()
IRIS_MATCH_THRESHOLD = 475  # Optimal threshold from the paper
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'
MODEL_PATH = os.path.join('model', 'iris_model_alexnet.h5')

# ===== ALEXNET MODEL =====
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ===== WEBCAM MANAGER =====
class WebcamManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            with self.lock:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.latest_frame = frame
            time.sleep(0.03)

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def is_active(self):
        return self.cap.isOpened()

webcam = WebcamManager()

# ===== IRIS PROCESSOR (IMPLEMENTING PAPER METHODS) =====
# Tambahkan kembali fungsi calculate_distance ke dalam IrisProcessor
class IrisProcessor:
    def preprocess_image(self, img):
        return cv2.cvtColor(cv2.resize(img, (776, 437)), cv2.COLOR_BGR2GRAY)

    def detect_edges(self, img):
        edges = cv2.Canny(cv2.GaussianBlur(img, (5, 5), 0), 50, 150)
        if DEBUG_MODE:
            cv2.imwrite("debug_edges.jpg", edges)
        return edges

    def detect_pupil(self, img):
        circles = cv2.HoughCircles(self.detect_edges(img), cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=5, maxRadius=80)
        if circles is not None:
            return tuple(map(int, circles[0][0]))
        return None

    def detect_iris(self, img, pupil):
        if pupil is None: return None
        x, y, _ = pupil
        circles = cv2.HoughCircles(self.detect_edges(img), cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=60, maxRadius=200)
        if circles is None: return None
        return min((tuple(map(int, c)) for c in circles[0]), key=lambda c: math.hypot(c[0]-x, c[1]-y))

    def normalize_iris(self, img, pupil, iris):
        if not pupil or not iris: return None
        px, py, pr = pupil
        ix, iy, ir = iris
        height, width = 64, 256
        normalized = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                r = y / height
                theta = 2 * np.pi * x / width
                radius = pr + r * (ir - pr)
                src_x, src_y = int(px + radius * math.cos(theta)), int(py + radius * math.sin(theta))
                if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                    normalized[y, x] = img[src_y, src_x]
        return normalized

    def gabor_filter(self, img, freq, theta):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getGaborKernel((35, 35), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        return cv2.filter2D(img, cv2.CV_8UC3, kernel)

    def average_absolute_deviation(self, region):
        mean = np.mean(region)
        return np.mean(np.abs(region - mean))

    def extract_features(self, normalized):
        if normalized is None: return None, None
        features = []
        for theta in map(np.deg2rad, [0, 45, 90, 135]):
            filtered = self.gabor_filter(normalized, 0.1, theta)
            for i in range(8):
                for j in range(8):
                    region = filtered[i*8:(i+1)*8, j*32:(j+1)*32]
                    features.append(self.average_absolute_deviation(region))
        return np.array(features, dtype=np.float32).tobytes(), np.ones(len(features), dtype=np.uint8).tobytes()

    def calculate_distance(self, f1, f2):
        if not f1 or not f2:
            return float('inf')
        a = np.frombuffer(f1, dtype=np.float32)
        b = np.frombuffer(f2, dtype=np.float32)
        return np.linalg.norm(a - b)

iris_processor = IrisProcessor()

# ===== UTILITY FUNCTIONS =====
def predict_eye_noeye(img):
    try:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        return "eye" if np.argmax(predictions) == 0 else "noeye"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "error"

def get_db_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset=MYSQL_CHARSET,
        cursorclass=pymysql.cursors.DictCursor
    )

def encrypt_data(data):
    if not isinstance(data, bytes):
        data = data.encode()
    cipher = AES.new(AES_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return cipher.nonce + tag + ciphertext

def decrypt_data(encrypted_data):
    if len(encrypted_data) < 32:
        raise ValueError("Encrypted data too short")
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(AES_KEY, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def log_audit(event, username=None, status=None):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            sql = "INSERT INTO audit_log (event, username, status) VALUES (%s, %s, %s)"
            cursor.execute(sql, (event, username, status))
        connection.commit()
    except Exception as e:
        print(f"Failed to log audit: {e}")
    finally:
        if connection:
            connection.close()

def init_db():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS iris_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(80) UNIQUE NOT NULL,
                    iris_template LONGBLOB NOT NULL,
                    iris_mask LONGBLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event VARCHAR(100) NOT NULL,
                    username VARCHAR(80),
                    status VARCHAR(255)
                )
            """)
        connection.commit()
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if connection:
            connection.close()

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = webcam.get_frame()
            if frame is None:
                continue

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan', methods=['POST'])
def scan():
    img = webcam.get_frame()
    if img is None:
        return jsonify({"status": "error", "message": "Gagal mengambil gambar"})

    if predict_eye_noeye(img) != "eye":
        return jsonify({"status": "noeye", "message": "Iris tidak terdeteksi"})

    gray = iris_processor.preprocess_image(img)
    pupil = iris_processor.detect_pupil(gray)
    iris = iris_processor.detect_iris(gray, pupil)
    if pupil is None or iris is None:
        return jsonify({"status": "error", "message": "Segmentasi gagal"})

    normalized = iris_processor.normalize_iris(img, pupil, iris)
    template_bytes, _ = iris_processor.extract_features(normalized)

    if not template_bytes:
        return jsonify({"status": "error", "message": "Gagal mengekstrak template iris"})

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT username, iris_template FROM iris_data")
            best_match = None
            min_distance = float('inf')

            for row in cursor.fetchall():
                try:
                    decrypted_template = decrypt_data(row['iris_template'])
                    distance = iris_processor.calculate_distance(template_bytes, decrypted_template)
                    print(f"[DEBUG] Distance to {row['username']}: {distance}")
                    if distance < min_distance:
                        min_distance = distance
                        best_match = row['username']
                except:
                    continue

            if min_distance < 475:
                log_audit("SCAN_SUCCESS", username=best_match, status=f"distance={min_distance}")
                return jsonify({"status": "match", "username": best_match, "message": f"Akses diberikan kepada {best_match}"})
            else:
                log_audit("SCAN_FAIL", status=f"Jarak minimum={min_distance}")
                return jsonify({"status": "no_match", "message": "User tidak dikenali"})

@app.route('/enroll', methods=['POST'])
def enroll_user():

    connection = None

    username = request.form.get('username')
    if not username:
        return jsonify({
            "status": "error",
            "message": "Username diperlukan"
        })

    image = webcam.get_frame()
    if image is None:
        return jsonify({
            "status": "error",
            "message": "Gagal mengambil gambar dari kamera"
        })

    try:
        # Preprocessing
        processed_img = iris_processor.preprocess_image(image)
        
        # Pupil detection
        pupil_circle = iris_processor.detect_pupil(processed_img)
        if pupil_circle is None:
            raise Exception("Pupil tidak terdeteksi")

        # Iris detection
        iris_circle = iris_processor.detect_iris(processed_img, pupil_circle)
        if iris_circle is None:
            raise Exception("Batas iris tidak terdeteksi")

        # Normalization
        normalized_iris = iris_processor.normalize_iris(image, pupil_circle, iris_circle)
        
        # Feature extraction
        template_bytes, mask_bytes = iris_processor.extract_features(normalized_iris)

        # Encrypt and save to database
        encrypted_template = encrypt_data(template_bytes)
        encrypted_mask = encrypt_data(mask_bytes)

        connection = get_db_connection()
        with connection.cursor() as cursor:
            # Check for existing similar iris
            cursor.execute("SELECT username, iris_template FROM iris_data")
            for row in cursor.fetchall():
                try:
                    decrypted_template = decrypt_data(row['iris_template'])
                    distance = iris_processor.calculate_euclidean_distance(
                        template_bytes,
                        decrypted_template
                    )
                    if distance < IRIS_MATCH_THRESHOLD:
                        return jsonify({
                            "status": "error",
                            "message": f"Iris sudah terdaftar sebagai '{row['username']}'"
                        })
                except Exception as e:
                    continue

            # Save new user
            cursor.execute(
                "INSERT INTO iris_data (username, iris_template, iris_mask) VALUES (%s, %s, %s)",
                (username, encrypted_template, encrypted_mask)
            )
            connection.commit()
            
            return jsonify({
                "status": "success",
                "username": username,
                "message": "Pendaftaran berhasil"
            })

    except pymysql.err.IntegrityError:
        return jsonify({
            "status": "error",
            "message": "Username sudah terdaftar"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Gagal mendaftarkan pengguna: {str(e)}"
        })
    finally:
        if connection:
            connection.close()

@app.route('/audit_logs')
def get_audit_logs():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 50")
            logs = cursor.fetchall()
            return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        if connection:
            connection.close()

@app.route('/admin')
def admin_dashboard():
    return render_template('admin.html')

@app.route('/admin/users')
def get_users():
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, username, created_at FROM iris_data")
            users = cursor.fetchall()
            return jsonify({"users": users})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        if connection:
            connection.close()

@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute("SELECT username FROM iris_data WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            
            if user:
                username = user['username']
                cursor.execute("DELETE FROM iris_data WHERE id = %s", (user_id,))
                connection.commit()
                log_audit("USER_DELETED", username=username, status="Admin action")
                return jsonify({"status": "success", "message": "User deleted"})
            else:
                return jsonify({"status": "error", "message": "User not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    finally:
        if connection:
            connection.close()

# Tambahan route baru untuk /register
@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/do_register', methods=['POST'])
def do_register():
    username = request.form.get('username')
    if not username:
        return jsonify({"status": "error", "message": "Username wajib diisi"})

    img = webcam.get_frame()
    if img is None:
        return jsonify({"status": "error", "message": "Gagal mengambil gambar"})

    gray = iris_processor.preprocess_image(img)
    pupil = iris_processor.detect_pupil(gray)
    iris = iris_processor.detect_iris(gray, pupil)
    if pupil is None or iris is None:
        return jsonify({"status": "error", "message": "Segmentasi iris gagal, mohon ulangi."})

    if iris[2] - pupil[2] < 5:
        print(f"[DEBUG] Pupil: {pupil}, Iris: {iris}")
        return jsonify({"status": "error", "message": "Jarak pupil dan iris terlalu kecil"})

    normalized = iris_processor.normalize_iris(img, pupil, iris)
    template_bytes, mask_bytes = iris_processor.extract_features(normalized)

    if not template_bytes:
        return jsonify({"status": "error", "message": "Template iris gagal diproses"})

    # Validasi stabilitas template
    self_distance = iris_processor.calculate_distance(template_bytes, template_bytes)
    print(f"[DEBUG] Self distance: {self_distance}")
    if self_distance > 20:
        return jsonify({"status": "error", "message": "Scan tidak stabil, harap ulangi."})

    encrypted_template = encrypt_data(template_bytes)
    encrypted_mask = encrypt_data(mask_bytes)

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Cek username duplikat
            cursor.execute("SELECT username FROM iris_data WHERE username = %s", (username,))
            if cursor.fetchone():
                return jsonify({"status": "error", "message": f"Username '{username}' sudah terdaftar"})

            # Cek kemiripan iris
            cursor.execute("SELECT username, iris_template FROM iris_data")
            for row in cursor.fetchall():
                try:
                    decrypted_template = decrypt_data(row['iris_template'])
                    distance = iris_processor.calculate_distance(template_bytes, decrypted_template)
                    print(f"[DEBUG] Distance to {row['username']}: {distance}")

                    if distance < 475:
                        log_audit("REGISTER_FAIL_DUPLICATE", username=username, status=f"Mirip {row['username']} distance={distance}")
                        return jsonify({"status": "error", "message": f"Iris sudah terdaftar sebagai '{row['username']}'"})
                    elif distance < 650:
                        log_audit("REGISTER_SUSPICIOUS_DUPLICATE", username=username, status=f"Terlalu mirip {row['username']} distance={distance}")
                        return jsonify({"status": "error", "message": f"Iris terlalu mirip dengan '{row['username']}', mohon pastikan ini pengguna berbeda."})
                except Exception as e:
                    continue

            cursor.execute("INSERT INTO iris_data (username, iris_template, iris_mask) VALUES (%s, %s, %s)", (username, encrypted_template, encrypted_mask))
            conn.commit()
            log_audit("USER_REGISTERED", username=username, status="success")
            return jsonify({"status": "success", "message": "Pendaftaran berhasil", "username": username})

@app.route('/system_status')
def system_status():
    status = {
        "webcam": webcam.is_active(),
        "database": "OK",
        "model_loaded": model is not None,
        "last_scan": datetime.now().isoformat()
    }
    
    try:
        connection = get_db_connection()
        connection.ping()
    except Exception as e:
        status["database"] = f"Error: {str(e)}"
    finally:
        if connection:
            connection.close()
            
    return jsonify(status)

@atexit.register
def cleanup():
    print("[INFO] Menutup webcam...")
    webcam.stop()

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=DEBUG_MODE)