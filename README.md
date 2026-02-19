# yolov5-drowsiness-detection
Real-time drowsiness detection using custom-trained YOLOv5 model.

##  Project Overview

This project implements a real-time drowsiness detection system using a custom-trained YOLOv5 object detection model. The system classifies webcam images into two states:

- Awake
- Drowsy

The model performs live detection through a webcam feed, making it suitable for driver monitoring systems, fatigue detection, and safety applications.

--- # installing requirements
--- !pip install -r yolov5/requirements.txt
--- !pip install torch torchvision torchaudio --upgrade

## 🎯 Objectives

- Build a computer vision model for detecting drowsiness.
- Collect custom image dataset using webcam
- Annotate dataset using LabelImg
- Train YOLOv5 custom detector
- Deploy model for real-time webcam inference


## 🧠 Tech Stack

- Python
- PyTorch
- YOLOv5
- OpenCV
- NumPy
- Matplotlib
## installing and importing dependencies
--- import os
--- os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
--- #confirming the dependencies
--- import torch
--- import torchaudio
--- import torchvision
--- print(f"Torch: {torch.__version__}")
--- print(f"Audio: {torchaudio.__version__}")
--- print(f"Vision: {torchvision.__version__}")
--- # This will return True if they can actually talk to each other
--- print("Is CUDA available?:", torch.cuda.is_available())

## 📂 Dataset Creation Process

1. Captured images using webcam
2. Created two classes:
   - awake
   - drowsy
3. Stored images inside: data/images/
4. Used LabelImg for bounding-box annotation
5. Generated YOLO-format labels

--- from matplotlib import pyplot as plt
--- import numpy as np
--- import cv2

## Load the model
--- model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

## 🏋️ Model Training
# making detections with images
--- img = 'https://c7.alamy.com/comp/HJMGJT/busy-city-road-traffic-in-nottingham-england-uk-HJMGJT.jpg'

--- results = model(img)
--- results.print

--- %matplotlib inline
--- plt.imshow(np.squeeze(results.render()))
--- plt.show()

## Real time detection

--- cap = cv2.VideoCapture(0)
--- while cap.isOpened():
    --- ret, frame = cap.read()

    --- # make detection
    --- results = model(frame)
    --- cv2.imshow('YOLO', np.squeeze(results.render()))

    --- if cv2.waitKey(10) & 0xFF == ord('q'):
        --- break    
--- cap.release()
--- cv2.destoryAllWindows()

Training was performed using YOLOv5 with:
python train.py --img 320 --batch 16 --epochs 200 --data dataset.yaml --weights yolov5s.pt

## Training a model from scratch
--- import uuid
--- import time

--- IMAGE_PATH = os.path.join('data', 'images')
--- labels = ['awake', 'drowsy']
--- num_imgs = 20

Frames are passed to the YOLO model, and predictions are rendered live.

Press **Q** to exit the detection window.

--- cap = cv2.VideoCapture(0)
--- #loop through lables
--- for label in labels:
    --- print(f'Collecting images for {label}')
    --- time.sleep(5)

    --- #loop through image range
    --- for img_num in range(num_imgs):
        --- print(f'Collecting images for {label}, imgage number {img_num}')
        --- # webcam feed
        --- ret, frame = cap.read()

        --- #naming out image path
        --- imgname = os.path.join(IMAGE_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        --- #writes out image to file
        --- cv2.imwrite(imgname, frame)
        --- #Render to the screen
        --- cv2.imshow('Image Collection', frame)
        
        --- # 2seconds delay between captures
        --- time.sleep(2)

        --- if cv2.waitKey(10) & 0xFF == ord('q'):
           ---  break    
--- cap.release()
--- cv2.destroyAllWindows()

--- !git clone https://github.com/tzutalin/labelImg
--- !pip install pyqt5 lxml --upgrade
--- !cd labelImg && pyrcc5 -o libs/resources.py resources.qrc

## 🔥 Key Features

✔ Custom dataset collection pipeline  
✔ Manual annotation workflow  
✔ YOLOv5 transfer learning  
✔ Real-time webcam detection  
✔ End-to-end deep learning pipeline 

## Loading the mode
--- !cd yolov5 && python train.py --img 320 --batch 16 --epochs 200 --data dataset.yaml --weights yolov5s.pt --workers 2
--- model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'yolov5/runs/train/exp7/weights/last.pt')

## Testing the results
--- img = os.path.join('data', 'images', 'awake.ead1e392-00fd-11f1-be03-e4b31815409a.jpg')
--- results = model(img)
--- results.print

#printing the result

--- %matplotlib inline
--- plt.imshow(np.squeeze(results.render()))
--- plt.show()

## 🚀 Future Improvements

- Add eye-aspect-ratio based detection
- Increase dataset size for better accuracy
- Deploy as desktop application
- Add alarm trigger for drowsy detection
- Convert model to ONNX for edge deployment
  
## Real time detection of my model
--- cap = cv2.VideoCapture(0)
--- while cap.isOpened():
    --- ret, frame = cap.read()

    --- #make detection
    --- results = model(frame)
    --- cv2.imshow('YOLO', np.squeeze(results.render()))

    --- if cv2.waitKey(10) & 0xFF == ord('q'):
       --- break    
--- cap.release()
--- cv2.destroyAllWindows()

## 👨‍💻 Author

Joseph  
Aspiring Data Scientist / Machine Learning Engineer





