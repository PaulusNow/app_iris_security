import cv2
import threading
import time

class WebcamManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None

        # Mulai thread untuk terus membaca frame
        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            with self.lock:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.latest_frame = frame
            time.sleep(0.03)  # ~30 FPS

    def get_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
