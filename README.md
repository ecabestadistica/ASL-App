# ASL Recognition - Hybrid Model System

Real-time American Sign Language (ASL) alphabet recognition using a hybrid deep learning approach.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-CC_BY_NC_4.0-green)

## Overview

This application recognizes the 24 static ASL fingerspelling letters (A-Y, excluding J and Z, which require motion) with a landmark-aware adaptive system that combines two DenseNet models:

- **DenseNet169** (landmark model): trained on images of the MediaPipe hand skeleton drawn on a black background (first 53 layers frozen). It is used whenever MediaPipe detects hand landmarks.
- **DenseNet201** (image model): trained on the original hand images (all layers fine-tuned from ImageNet weights). It is used when no landmarks are detected.

Two modes are available:

- `switching` (default): only one network is run per frame, following the decision rule of the paper.
- `late_fusion`: when landmarks are detected, the softmax outputs of both networks are averaged (more accurate, about half the frame rate).

## Features

- 🎥 Real-time webcam hand detection using MediaPipe
- 🤖 Landmark-aware adaptive recognition (switching or late fusion)
- 📊 Top-5 predictions with confidence scores
- 📚 "Letter of the Day" learning feature with example images
- 🌐 Web-based interface using Flask

## Demo video

![Demo del proyecto](Demo.gif)

## Installation

### Prerequisites

- Python 3.10
- Webcam
- macOS, Linux, or Windows

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ecabestadistica/ASL-App.git
   cd ASL-App
   ```

2. **Create a conda environment** (recommended)
   ```bash
   conda create -n asl python=3.10
   conda activate asl
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the models**
   
   Download the model files from Google Drive and place them in the project root:
   
   📥 **[Download Models](https://drive.google.com/drive/folders/1DiIeXnnWyvxBmG-gLtG7Cj1Col5MnYTz?usp=drive_link)**
   
   - `landmark_model_SIG79.h5` (67 MB) - DenseNet169 trained on landmark-only images
   - `image_model_AL94.h5` (92 MB) - DenseNet201 trained on the original images

5. **Example images (learning mode)** 
   
   Make sure that you have the `Examples/` folder with example images for each letter:
   ```
   Examples/
   ├── A.webp
   ├── B.webp
   └── ...
   ```

## Usage

1. **Start the application**
   ```bash
   python app_hybrid.py                      # webcam, switching mode
   python app_hybrid.py --mode late_fusion   # webcam, late fusion
   python app_hybrid.py --source video.mp4   # a video file instead of the webcam
   ```
   The mode can also be changed while the app is running: `http://localhost:5000/set_mode?mode=late_fusion`.

2. **Open your browser**
   
   Navigate to `http://localhost:5000`

3. **Start signing!**
   - Position your hand clearly in the camera view
   - Make ASL signs (follow the examples to learn)
   - The system will display predictions in real-time

## Project Structure

```
ASL-App/
├── app_hybrid.py          # Main Flask application
├── templates/
│   └── index.html         # Web interface
├── Examples/              # Example images for learning
│   ├── A.webp
│   └── ...
├── landmark_model_SIG79.h5  # DenseNet169 landmark model (not included in repo: download from Drive)
├── image_model_AL94.h5      # DenseNet201 image model (not included in repo: download from Drive)
├── requirements.txt       # Python dependencies
└── README.md
```

## How It Works

### Landmark-aware adaptive system

1. **Hand detection**: MediaPipe Hands (tracking mode) detects the hand landmarks in each frame.
2. **Input preparation** (as during training: RGB, 192x192, values in [0, 1]):
   - Landmark model: the hand skeleton drawn with the default MediaPipe style on a black 200x200 canvas of a square crop centred on the hand.
   - Image model: the same hand crop or, when no landmarks are detected, the central square of the frame.
3. **Decision rule**:
   - `switching`: landmarks detected → DenseNet169; otherwise → DenseNet201.
   - `late_fusion`: landmarks detected → average of DenseNet169 and DenseNet201; otherwise → DenseNet201.
4. **Temporal smoothing**: the final prediction is the mode of the last 10 frames.

Measured on CPU (Apple M5 Pro, no GPU): about 23 FPS in `switching` mode and about 12 FPS in `late_fusion` mode.
The web interface polls the latest prediction every 100 ms.

### Model Details

| Model | Architecture | Input | Training Data |
|-------|-------------|-------|---------------|
| Landmark | DenseNet169 (first 53 layers frozen) | 192x192 landmark image | Landmark-only images |
| Image | DenseNet201 (all layers fine-tuned) | 192x192 hand image | Original images |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream |
| `/predictions` | GET | Current predictions, model used, mode and timings (JSON) |
| `/stats` | GET | Processing FPS and median time per stage (JSON) |
| `/set_mode?mode=...` | GET | Switch between `switching` and `late_fusion` |
| `/change_letter` | GET | Change letter of the day |

## Requirements

- TensorFlow 2.15.0
- MediaPipe
- OpenCV
- Flask
- NumPy

See `requirements.txt` for complete list.

## Troubleshooting

### Camera not detected
```bash
# Try different camera indices in the code (0, 1, 2)
```

### Model loading errors
- Ensure you're using TensorFlow 2.15.0 (not 2.16+)
- Models must be in `.h5` format

### MediaPipe errors
```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the CC BY-NC 4.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand detection
- [TensorFlow](https://www.tensorflow.org/) for deep learning framework
- ASL alphabet dataset contributors

## Author

**Elisa Cabana, CUNEF Universidad, Madrid, Spain elisa.cabana@cunef.edu**

---

⭐ If you found this project useful, please consider giving it a star!
