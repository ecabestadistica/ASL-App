"""
ASL Recognition - Hybrid Model
Uses keypoints model DenseNet169 when landmarks detected and DenseNet201 on cropped hand image

Compatible with index.html template
Requires: TensorFlow 2.15.0
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter, deque
import random
import os
import mediapipe as mp
import base64

app = Flask(__name__)

# Configuration
KEYPOINTS_MODEL_PATH = 'keypoints_model.h5'  # DenseNet169 trained on keypoint visualizations
FINAL_MODEL_PATH = 'final_model.h5'          # DenseNet201 trained on hand images
EXAMPLES_FOLDER = 'Examples'
IMG_SIZE = 192
LETTERS = [l for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if l not in ['J', 'Z']]
FRAMES_FOR_PREDICTION = 10


class HybridASLRecognizer:
    def __init__(self):
        print("="*60)
        print("Loading Hybrid ASL Recognition System")
        print("="*60)
        
        # Load keypoints model (primary)
        print("\n[1/2] Loading keypoints model (DenseNet169)...")
        try:
            self.keypoints_model = keras.models.load_model(KEYPOINTS_MODEL_PATH, compile=False)
            print(f"  ✓ Keypoints model loaded")
            print(f"    Input shape: {self.keypoints_model.input_shape}")
            self.has_keypoints_model = True
        except Exception as e:
            print(f"  ✗ Could not load keypoints model: {e}")
            self.keypoints_model = None
            self.has_keypoints_model = False
        
        # Load final model (fallback)
        print("\n[2/2] Loading final model (DenseNet201)...")
        try:
            self.final_model = keras.models.load_model(FINAL_MODEL_PATH, compile=False)
            print(f"  ✓ Final model loaded")
            print(f"    Input shape: {self.final_model.input_shape}")
            self.has_final_model = True
        except Exception as e:
            print(f"  ✗ Could not load final model: {e}")
            self.final_model = None
            self.has_final_model = False
        
        if not self.has_keypoints_model and not self.has_final_model:
            raise RuntimeError("No models could be loaded!")
        
        # Initialize MediaPipe
        print("\nInitializing MediaPipe...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        print("  ✓ MediaPipe ready")
        
        # Load example images
        self.example_images = self._load_examples()
        
        # State variables
        self.prediction_buffer = deque(maxlen=FRAMES_FOR_PREDICTION)
        self.frame_count = 0
        self.final_prediction = ""
        self.current_top5 = []
        self.current_probs = []
        self.hand_detected = False
        self.current_model_used = None
        
        # Letter of the day
        available = list(self.example_images.keys())
        self.letter_of_day = random.choice(available) if available else 'A'
        
        # Open camera
        print("\nOpening camera...")
        self.cap = None
        for idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    print(f"  ✓ Camera opened (index {idx})")
                    break
                self.cap.release()
        
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"\n{'='*60}")
        print(f"System ready!")
        print(f"  - Keypoints model: {'✓' if self.has_keypoints_model else '✗'}")
        print(f"  - Final model: {'✓' if self.has_final_model else '✗'}")
        print(f"  - Example images: {len(self.example_images)}")
        print(f"  - Letter of the day: {self.letter_of_day}")
        print(f"{'='*60}\n")
    
    def _load_examples(self):
        """Load example images - supports both flat and subfolder structures"""
        examples = {}
        
        if not os.path.exists(EXAMPLES_FOLDER):
            print(f"  ⚠ Examples folder '{EXAMPLES_FOLDER}' not found")
            return examples
        
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for item in os.listdir(EXAMPLES_FOLDER):
            item_path = os.path.join(EXAMPLES_FOLDER, item)
            
            if os.path.isdir(item_path):
                # Subfolder structure: Examples/A/image.png
                letter = item.upper()
                if letter in LETTERS:
                    images = [f for f in os.listdir(item_path) 
                             if os.path.splitext(f)[1].lower() in valid_ext]
                    if images:
                        examples[letter] = os.path.join(item_path, random.choice(images))
            else:
                # Flat structure: Examples/A.png
                ext = os.path.splitext(item)[1].lower()
                if ext in valid_ext:
                    name = os.path.splitext(item)[0].upper()
                    for char in name:
                        if char in LETTERS:
                            examples[char] = item_path
                            break
        
        print(f"  ✓ {len(examples)} example images loaded")
        return examples
    
    def create_keypoint_image(self, landmarks, frame_shape):
        """Create keypoint visualization on black background"""
        canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        points = []
        for lm in landmarks.landmark:
            x = int(lm.x * IMG_SIZE)
            y = int(lm.y * IMG_SIZE)
            points.append((x, y))
        
        # Draw connections (gray)
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(canvas, points[start_idx], points[end_idx], (128, 128, 128), 2)
        
        # Draw landmarks (red)
        for point in points:
            cv2.circle(canvas, point, 4, (0, 0, 255), -1)
        
        return canvas
    
    def get_hand_bbox(self, landmarks, frame_shape):
        """Get bounding box from landmarks"""
        h, w = frame_shape[:2]
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        
        # Add 20% margin
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        # Make square
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = max(0, center_x - size // 2)
        y_min = max(0, center_y - size // 2)
        x_max = min(w, x_min + size)
        y_max = min(h, y_min + size)
        
        if x_max - x_min < size:
            x_min = max(0, x_max - size)
        if y_max - y_min < size:
            y_min = max(0, y_max - size)
        
        return (x_min, y_min, x_max, y_max)
    
    def predict_with_model(self, model, image):
        """Run prediction with a model, return top 5"""
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        predictions = model.predict(img, verbose=0)[0]
        
        top_5_idx = np.argsort(predictions)[-5:][::-1]
        top_5_labels = [LETTERS[i] for i in top_5_idx]
        top_5_probs = [float(predictions[i]) for i in top_5_idx]
        
        return top_5_labels, top_5_probs
    
    def update_final_prediction(self, top_prediction):
        """Update final prediction using mode of buffer"""
        self.prediction_buffer.append(top_prediction)
        self.frame_count += 1
        
        if self.frame_count % FRAMES_FOR_PREDICTION == 0 and len(self.prediction_buffer) > 0:
            counter = Counter(self.prediction_buffer)
            self.final_prediction = counter.most_common(1)[0][0]
    
    def get_frame(self):
        """Process frame and return encoded image with predictions"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Detect hand
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            self.hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            bbox = self.get_hand_bbox(hand_landmarks, frame.shape)
            
            # Try keypoints model first (primary)
            keypoints_labels, keypoints_probs = None, None
            if self.has_keypoints_model:
                keypoint_image = self.create_keypoint_image(hand_landmarks, frame.shape)
                keypoints_labels, keypoints_probs = self.predict_with_model(
                    self.keypoints_model, keypoint_image
                )
            
            # Try final model (fallback or comparison)
            final_labels, final_probs = None, None
            if self.has_final_model:
                x_min, y_min, x_max, y_max = bbox
                hand_roi = frame[y_min:y_max, x_min:x_max]
                if hand_roi.size > 0:
                    final_labels, final_probs = self.predict_with_model(
                        self.final_model, hand_roi
                    )
            
            # Decide which model to use
            if keypoints_labels and keypoints_probs:
                if final_labels and final_probs and final_probs[0] > keypoints_probs[0]:
                    # Final model more confident
                    self.current_top5 = final_labels
                    self.current_probs = final_probs
                    self.current_model_used = "final"
                else:
                    # Keypoints model
                    self.current_top5 = keypoints_labels
                    self.current_probs = keypoints_probs
                    self.current_model_used = "keypoints"
            elif final_labels:
                self.current_top5 = final_labels
                self.current_probs = final_probs
                self.current_model_used = "final"
            
            if self.current_top5:
                self.update_final_prediction(self.current_top5[0])
            
            # Draw bounding box (always green)
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
        else:
            self.hand_detected = False
            self.current_top5 = []
            self.current_probs = []
            self.current_model_used = None
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    def get_example_image_base64(self, letter):
        """Get example image as base64 string"""
        if letter not in self.example_images:
            return None
        
        img_path = self.example_images[letter]
        try:
            with open(img_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Detect extension for mime type
                ext = os.path.splitext(img_path)[1].lower()
                if ext == '.webp':
                    return f"data:image/webp;base64,{img_base64}"
                elif ext in ['.jpg', '.jpeg']:
                    return f"data:image/jpeg;base64,{img_base64}"
                else:
                    return f"data:image/png;base64,{img_base64}"
        except:
            return None
    
    def get_hand_status(self):
        """Get hand detection status for UI"""
        if self.hand_detected:
            return "detected"
        elif len(self.prediction_buffer) > 0:
            return "lost"
        else:
            return "waiting"


# Global recognizer
recognizer = None


def get_recognizer():
    global recognizer
    if recognizer is None:
        recognizer = HybridASLRecognizer()
    return recognizer


def gen_frames():
    """Generate video frames for streaming"""
    rec = get_recognizer()
    while True:
        frame_bytes = rec.get_frame()
        if frame_bytes is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """Serve main page - uses existing index.html template"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predictions')
def get_predictions():
    """Get current predictions - compatible with existing template"""
    rec = get_recognizer()
    
    # Get example image for letter of the day
    example_img_base64 = rec.get_example_image_base64(rec.letter_of_day)
    
    data = {
        'final_prediction': rec.final_prediction,
        'top5_labels': rec.current_top5,
        'top5_probs': rec.current_probs,
        'letter_of_day': rec.letter_of_day,
        'example_image': example_img_base64,
        'hand_status': rec.get_hand_status(),
        'model_used': rec.current_model_used  # Extra info for hybrid
    }
    return jsonify(data)


@app.route('/change_letter')
def change_letter():
    """Change the letter of the day"""
    rec = get_recognizer()
    available = list(rec.example_images.keys())
    if available:
        rec.letter_of_day = random.choice(available)
    else:
        rec.letter_of_day = random.choice(LETTERS)
    return jsonify({'letter': rec.letter_of_day})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤟 ASL Recognition - Hybrid Model System")
    print("="*60)
    
    # Initialize recognizer
    recognizer = HybridASLRecognizer()
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=False, threaded=True)
