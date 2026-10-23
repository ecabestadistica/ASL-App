"""
ASL Recognition - Landmark-aware adaptive system

Implements the decision rule of the paper:
    f(I) = f_D169(phi(I))  if MediaPipe detects hand landmarks
           f_D201(I)       otherwise
where f_D169 is DenseNet169 trained on landmark-only images (Freeze 53, SIG-79) and
f_D201 is DenseNet201 trained on the original images (Train All, AL-94).

Two modes:
  - switching   (default): only one network is run per frame (the paper's rule).
  - late_fusion : when landmarks are detected, the softmax outputs of both networks are averaged;
                  otherwise DenseNet201 alone is used.

Inputs are prepared exactly as during training: RGB, 192x192, values in [0, 1]; the landmark image is
the MediaPipe skeleton drawn with the default style on a black 200x200 canvas of a hand-centred crop.

Requires: TensorFlow 2.15.0, MediaPipe 0.10.9
"""

import argparse
import base64
import os
import random
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, render_template, request
from mediapipe.framework.formats import landmark_pb2

app = Flask(__name__)

# Configuration
LANDMARK_MODEL_PATH = 'landmark_model_SIG79.h5'  # DenseNet169, landmark-only images, first 53 layers frozen
IMAGE_MODEL_PATH = 'image_model_AL94.h5'         # DenseNet201, original images, all layers trained
EXAMPLES_FOLDER = 'Examples'
IMG_SIZE = 192                 # network input size
CANVAS_SIZE = 200              # size of the landmark images used for training
CROP_SCALE = 2.1               # crop side / hand size; in the training images the hand spans ~47% of the image
LETTERS = [l for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if l not in ['J', 'Z']]
FRAMES_FOR_PREDICTION = 10     # final prediction = mode of the last 10 frames
MODES = ('switching', 'late_fusion')


class HybridASLRecognizer:
    def __init__(self, mode='switching', source=None):
        print("=" * 60)
        print("Loading ASL recognition system")
        print("=" * 60)

        self.landmark_model = tf.keras.models.load_model(LANDMARK_MODEL_PATH, compile=False)
        self.image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
        # Compiled calls are much faster than model.predict() for single frames
        self._predict_landmark = tf.function(lambda x: self.landmark_model(x, training=False))
        self._predict_image = tf.function(lambda x: self.image_model(x, training=False))
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), np.float32)
        self._predict_landmark(dummy)
        self._predict_image(dummy)
        print("  ✓ Landmark model (DenseNet169) and image model (DenseNet201) loaded")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mode = mode
        self.lock = threading.Lock()
        self.prediction_buffer = deque(maxlen=FRAMES_FOR_PREDICTION)
        self.final_prediction = ""
        self.current_top5 = []
        self.current_probs = []
        self.hand_detected = False
        self.current_model_used = None
        self.timings = {k: deque(maxlen=100) for k in ('mediapipe', 'preprocessing', 'inference', 'total')}
        self.frame_times = deque(maxlen=100)

        self.example_images = self._load_examples()
        available = list(self.example_images.keys())
        self.letter_of_day = random.choice(available) if available else 'A'

        self.cap = self._open_source(source)
        print(f"\nSystem ready (mode: {self.mode})\n")

    # ------------------------------------------------------------------ setup
    def _open_source(self, source):
        """Webcam (default, first index that works) or a video file, which is played in a loop."""
        if source is not None and not str(source).isdigit():
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video {source}")
            self.loop_video = True
            return cap
        self.loop_video = False
        indices = [int(source)] if source is not None else [0, 1, 2]
        for idx in indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened() and cap.read()[0]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"  ✓ Camera opened (index {idx})")
                return cap
            cap.release()
        raise RuntimeError("Could not open camera")

    def _load_examples(self):
        """Example image per letter, already encoded as base64 so that it is not re-read on every request"""
        examples = {}
        if not os.path.exists(EXAMPLES_FOLDER):
            return examples
        mime = {'.webp': 'image/webp', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png'}
        for item in os.listdir(EXAMPLES_FOLDER):
            name, ext = os.path.splitext(item)
            if ext.lower() in mime and name.upper() in LETTERS:
                with open(os.path.join(EXAMPLES_FOLDER, item), 'rb') as f:
                    data = base64.b64encode(f.read()).decode('utf-8')
                examples[name.upper()] = f"data:{mime[ext.lower()]};base64,{data}"
        return examples

    # ------------------------------------------------------------- inputs
    @staticmethod
    def _to_input(img_rgb):
        """RGB uint8 image -> float tensor (1, 192, 192, 3) in [0, 1], resized as in training"""
        x = tf.image.resize(img_rgb[None].astype(np.float32), (IMG_SIZE, IMG_SIZE), method='bilinear')
        return x / 255.0

    @staticmethod
    def hand_crop_box(landmarks, w, h):
        """Square box centred on the hand, CROP_SCALE times the hand size (may extend beyond the frame)"""
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * CROP_SCALE
        side = max(side, 32)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        return int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(side))

    @staticmethod
    def crop_with_padding(img, x0, y0, side):
        """Crop a square that may fall partly outside the image, replicating the border"""
        h, w = img.shape[:2]
        pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
        if pad:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        return img[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]

    def landmark_canvas(self, landmarks, w, h, box):
        """Skeleton of the hand on a black CANVAS_SIZE canvas, re-normalised to the hand crop (as in training)"""
        x0, y0, side = box
        local = landmark_pb2.NormalizedLandmarkList()
        for lm in landmarks.landmark:
            local.landmark.add(x=(lm.x * w - x0) / side, y=(lm.y * h - y0) / side, z=lm.z)
        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        self.mp_draw.draw_landmarks(canvas, local, self.mp_hands.HAND_CONNECTIONS)  # default style, BGR
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------- inference
    def process_frame(self, frame_bgr):
        """Classify one frame. Returns the probabilities, the branch used, the landmarks and the timings (ms)."""
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        t1 = time.perf_counter()

        landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None
        if landmarks is not None:
            box = self.hand_crop_box(landmarks, w, h)
            x_landmark = self._to_input(self.landmark_canvas(landmarks, w, h, box))
            x_image = self._to_input(self.crop_with_padding(frame_rgb, *box)) if self.mode == 'late_fusion' else None
            t2 = time.perf_counter()
            probs = self._predict_landmark(x_landmark).numpy()[0]
            if x_image is not None:
                probs = (probs + self._predict_image(x_image).numpy()[0]) / 2
                model_used = 'fusion'
            else:
                model_used = 'landmarks'
        else:
            box = None
            side = min(h, w)  # no landmarks: central square of the frame
            x_image = self._to_input(frame_rgb[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2])
            t2 = time.perf_counter()
            probs = self._predict_image(x_image).numpy()[0]
            model_used = 'image'
        t3 = time.perf_counter()

        timings = {'mediapipe': (t1 - t0) * 1000, 'preprocessing': (t2 - t1) * 1000,
                   'inference': (t3 - t2) * 1000, 'total': (t3 - t0) * 1000}
        return probs, model_used, landmarks, box, timings

    def update_state(self, probs, model_used, landmarks, timings):
        top5 = np.argsort(probs)[-5:][::-1]
        with self.lock:
            self.current_top5 = [LETTERS[i] for i in top5]
            self.current_probs = [float(probs[i]) for i in top5]
            self.current_model_used = model_used
            self.hand_detected = landmarks is not None
            self.prediction_buffer.append(self.current_top5[0])
            self.final_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
            for k, v in timings.items():
                self.timings[k].append(v)
            self.frame_times.append(time.perf_counter())

    def get_frame(self):
        """Read a frame, classify it and return it as JPEG with the hand box and landmarks drawn"""
        ret, frame = self.cap.read()
        if not ret and self.loop_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret:
            return None
        if not self.loop_video:
            frame = cv2.flip(frame, 1)  # mirror view for the webcam

        probs, model_used, landmarks, box, timings = self.process_frame(frame)
        self.update_state(probs, model_used, landmarks, timings)

        if landmarks is not None:
            x0, y0, side = box
            cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 3)
            self.mp_draw.draw_landmarks(
                frame, landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    # ------------------------------------------------------------- state
    def get_stats(self):
        with self.lock:
            stats = {k: round(float(np.median(v)), 1) if v else None for k, v in self.timings.items()}
            fps = None
            if len(self.frame_times) > 1:
                fps = round((len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0]), 1)
        return {'mode': self.mode, 'processing_fps': fps, 'median_ms': stats}

    def get_hand_status(self):
        if self.hand_detected:
            return "detected"
        elif len(self.prediction_buffer) > 0:
            return "lost"
        return "waiting"


recognizer = None


def get_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = HybridASLRecognizer()
    return recognizer


def gen_frames():
    rec = get_recognizer()
    while True:
        frame_bytes = rec.get_frame()
        if frame_bytes is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
def get_predictions():
    rec = get_recognizer()
    with rec.lock:
        data = {
            'final_prediction': rec.final_prediction,
            'top5_labels': rec.current_top5,
            'top5_probs': rec.current_probs,
            'model_used': rec.current_model_used,
            'hand_status': rec.get_hand_status(),
        }
    data['letter_of_day'] = rec.letter_of_day
    data['example_image'] = rec.example_images.get(rec.letter_of_day)
    data.update(rec.get_stats())
    return jsonify(data)


@app.route('/stats')
def get_stats():
    return jsonify(get_recognizer().get_stats())


@app.route('/set_mode')
def set_mode():
    """Switch between 'switching' and 'late_fusion' at runtime, e.g. /set_mode?mode=late_fusion"""
    rec = get_recognizer()
    mode = request.args.get('mode', '')
    if mode in MODES:
        rec.mode = mode
    return jsonify({'mode': rec.mode})


@app.route('/change_letter')
def change_letter():
    rec = get_recognizer()
    available = list(rec.example_images.keys()) or LETTERS
    rec.letter_of_day = random.choice(available)
    return jsonify({'letter': rec.letter_of_day})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASL fingerspelling recognition (landmark-aware adaptive system)')
    parser.add_argument('--mode', choices=MODES, default='switching',
                        help="'switching' (paper rule, default) or 'late_fusion'")
    parser.add_argument('--source', default=None,
                        help='camera index (default: first available) or path to a video file')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    recognizer = HybridASLRecognizer(mode=args.mode, source=args.source)
    print(f"📱 Open http://localhost:{args.port} in your browser (Ctrl+C to stop)\n")
    app.run(debug=False, threaded=True, port=args.port)
