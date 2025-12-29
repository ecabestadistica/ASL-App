# ASL Recognition - Hybrid Model System

Real-time American Sign Language (ASL) alphabet recognition using a hybrid deep learning approach.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-CC_BY_NC_4.0-green)

## Overview

This application recognizes 24 static ASL letters (A-Y, excluding J and Z which require motion) using two DenseNet models in a hybrid configuration:

- **DenseNet169**: Primary model trained on MediaPipe hand landmark visualizations
- **DenseNet201**: Fallback model trained on cropped hand images

## Features

- 🎥 Real-time webcam hand detection using MediaPipe
- 🤖 Hybrid model architecture for improved accuracy
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
   
   📥 **[Download Models](https://drive.google.com/drive/folders/1m088EvFt1BiKPQkk-kV0--jS-srn6vn3?usp=sharing)**
   
   - `keypoints_model.h5` (67 MB) - DenseNet169 trained on keypoint visualizations
   - `final_model.h5` (127 MB) - DenseNet201 trained on hand images

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
   python app_hybrid.py
   ```

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
├── keypoints_model.h5     # DenseNet169 model (not included in repo: download from Drive)
├── final_model.h5         # DenseNet201 model (not included in repo: download from Drive)
├── requirements.txt       # Python dependencies
└── README.md
```

## How It Works

### Hybrid Model Architecture

1. **Hand Detection**: MediaPipe detects hand landmarks in real-time
2. **Dual Prediction**: 
   - Keypoints model receives a 192x192 black canvas with landmark visualization
   - Final model receives the cropped hand region from the original frame
3. **Prediction**: Hybrid prediction: if landmarks are detected: DenseNet169, if not: DenseNet201
4. **Temporal Smoothing**: Predictions are buffered over 10 frames to reduce noise

### Model Details

| Model | Architecture | Input | Training Data |
|-------|-------------|-------|---------------|
| Keypoints | DenseNet169 | 192x192 landmark visualization | Hand keypoint images |
| Final | DenseNet201 | 192x192 cropped hand | Natural hand images |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | MJPEG video stream |
| `/predictions` | GET | Current predictions (JSON) |
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
