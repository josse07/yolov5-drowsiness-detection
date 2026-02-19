# yolov5-drowsiness-detection
Real-time drowsiness detection using custom-trained YOLOv5 model

##  Project Overview

This project implements a real-time drowsiness detection system using a custom-trained YOLOv5 object detection model. The system classifies webcam images into two states:

- Awake
- Drowsy

The model performs live detection through a webcam feed, making it suitable for driver monitoring systems, fatigue detection, and safety applications.

--- 

## 🎯 Objectives

- Build a computer vision model for detecting drowsiness
- Collect custom image dataset using webcam
- Annotate dataset using LabelImg
- Train YOLOv5 custom detector
- Deploy model for real-time webcam inference

---

## 🧠 Tech Stack

- Python
- PyTorch
- YOLOv5
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Dataset Creation Process

1. Captured images using webcam
2. Created two classes:
   - awake
   - drowsy
3. Stored images inside: data/images/
4. Used LabelImg for bounding-box annotation
5. Generated YOLO-format labels

---

## 🏋️ Model Training

Training was performed using YOLOv5 with:
python train.py --img 320 --batch 16 --epochs 200 --data dataset.yaml --weights yolov5s.pt


Frames are passed to the YOLO model, and predictions are rendered live.

Press **Q** to exit the detection window.

---

## 🔥 Key Features

✔ Custom dataset collection pipeline  
✔ Manual annotation workflow  
✔ YOLOv5 transfer learning  
✔ Real-time webcam detection  
✔ End-to-end deep learning pipeline  

---

## 🚀 Future Improvements

- Add eye-aspect-ratio based detection
- Increase dataset size for better accuracy
- Deploy as desktop application
- Add alarm trigger for drowsy detection
- Convert model to ONNX for edge deployment

---

## 👨‍💻 Author

Joseph  
Aspiring Data Scientist / Machine Learning Engineer





