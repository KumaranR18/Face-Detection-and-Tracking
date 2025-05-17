<h1 align="center">🖼️🎥 Image & Video Face Detection and Tracking</h1>

Welcome to this robust and efficient OpenCV-based project for real-time **face detection** in both images and videos! Whether you want to detect faces in a single image or track multiple faces in a video stream, this project has you covered.

---

## ✨ Features

- Image Face Detection: Detect faces in any input image using Haar Cascade classifiers.
- Video Face Detection & Tracking: Detect and track faces frame-by-frame in videos or webcam feed.
- Real-time detection with clear bounding boxes highlighting detected faces.
- Easy to customize and extend for your own projects.

---

## 🧠 How It Works

- Uses OpenCV’s pre-trained Haar Cascade model (`haarcascade_frontalface_default.xml`) to detect faces.
- Processes grayscale images and videos to identify faces.
- Draws green rectangles around detected faces for visualization.

---

## 🚀 Usage

### 1️⃣ Image Face Detection

- Replace the `video_path` variable with the path to your image.
- Run the script, and a window will pop up showing detected faces.
- Press `ESC` to exit.

### 2️⃣ Video Face Detection & Tracking

- Replace the `video_path` variable with your video file path or use `0` for webcam.
- Processes video frame-by-frame, highlighting faces in real-time.
- Press `ESC` to stop and close the window.

---

## 🛠️ Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)

---

## 📥 Getting Started

- Clone the repo:
  ```bash
  git clone https://github.com/KumaranR18/Face-Detection-and-Tracking.git
  cd Face-Detection-and-Tracking
  ```
## Download the Haar Cascade XML file (included or from OpenCV GitHub):

[haarcascade_frontalface_default.xml](https://github.com/KumaranR18/Face-Detection-and-Tracking/blob/main/Face%20Detection%20and%20Tracking/haarcascade_frontalface_default.xml)

## Run the scripts:

1. For image detection:

``` bash
python Image Face Detection.py
```
2. For video detection:

```bash
python Video Face Detection and Tracking.py
```
## 🤝 Contributing

- Contributions welcome!
- Star the repo, report issues, or submit pull requests to improve detection or add features.
- Let’s build great face detection together!

---

## 📜 License

- Licensed under the **MIT License** — see the LICENSE file for details.
