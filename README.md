# Gender and Age Detection System# Gender-and-Age-Detection   <img alt="GitHub" src="https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection">



![Project Banner](frontend/logo1.png)

<h2>Objective :</h2>

A real-time AI surveillance system capable of detecting faces, estimating age and gender, identifying objects, and tracking movement paths.<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>



## 🤝 Sponsors<h2>About the Project :</h2>

<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

We are proud to be supported by:

<h2>Dataset :</h2>

<div align="center"><p>For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.</p>

  <img src="frontend/logo1.png" height="100" alt="Sponsor 1" style="background-color: white; padding: 20px; border-radius: 10px; margin: 10px;">

  <img src="frontend/logo4.png" height="100" alt="Sponsor 2" style="background-color: white; padding: 20px; border-radius: 10px; margin: 10px;"><h2>Additional Python Libraries Required :</h2>

</div><ul>

  <li>OpenCV</li>

## ✨ Features  

       pip install opencv-python

- **Real-time Face Detection**: Uses YuNet for high-speed face detection.</ul>

- **Age & Gender Estimation**: Caffe models provide accurate demographic estimates.<ul>

- **Person Re-identification**: SFace model recognizes returning visitors. <li>argparse</li>

- **Object Detection**: YOLOX-S model detects 80+ types of objects (people, cars, etc.).  
- **Advanced AI Analysis**: DeepFace integration for Emotion Recognition and Race Detection.
- **Liveness Detection**: Temporal analysis to verify live presence.
- **Path Tracking**: Visualizes the movement history of individuals.       pip install argparse
- **Visitor Database**: Tracks visit counts and timestamps for each unique person.</ul>

- **Dual Interface**:

  - **Live Feed**: Real-time camera view with augmented reality overlays.<h2>The contents of this Project :</h2>

  - **Dashboard**: Classy, modern interface for viewing visitor statistics.<ul>

  <li>opencv_face_detector.pbtxt</li>

## 🚀 Getting Started  <li>opencv_face_detector_uint8.pb</li>

  <li>age_deploy.prototxt</li>

### Prerequisites  <li>age_net.caffemodel</li>

  <li>gender_deploy.prototxt</li>

- Python 3.8+  <li>gender_net.caffemodel</li>

- OpenCV  <li>a few pictures to try the project on</li>

- FastAPI  <li>detect.py</li>

- Uvicorn </ul>

 <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>

### Installation 

 <h2>Usage :</h2>

1. Clone the repository <ul>

2. Install dependencies:  <li>Download my Repository</li>

   ```bash  <li>Open your Command Prompt or Terminal and change directory to the folder where all the files are present.</li>

   pip install fastapi uvicorn opencv-python numpy pillow  <li><b>Detecting Gender and Age of face in Image</b> Use Command :</li>

   ```  

3. Download the models (automatically handled by the start script or manually):      python detect.py --image <image_name>

   - `yolox_s.onnx`</ul>

   - `face_detection_yunet_2023mar.onnx`  <p><b>Note: </b>The Image should be present in same folder where all the files are present</p> 

   - `face_recognition_sface_2021dec.onnx`<ul>

  <li><b>Detecting Gender and Age of face through webcam</b> Use Command :</li>

### Running the Application  

      python detect.py

Simply run the start script:</ul>

<ul>

```bash  <li>Press <b>Ctrl + C</b> to stop the program execution.</li>

./start.sh</ul>

```

# Working:

This will launch:[![Watch the video](https://img.youtube.com/vi/ReeccRD21EU/0.jpg)](https://youtu.be/ReeccRD21EU)

- **Backend API** at `http://localhost:8000`

- **Frontend Interface** at `http://localhost:3000`<h2>Examples :</h2>

<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

## 🖥️ User Interface

    >python detect.py --image girl1.jpg

### Live Feed    Gender: Female

- **Cyan Box**: New Visitor    Age: 25-32 years

- **Green Box**: Returning Visitor    

- **Yellow Box**: Detected Objects<img src="Example/Detecting age and gender girl1.png">

- **Path Lines**: Movement history

    >python detect.py --image girl2.jpg

### Dashboard    Gender: Female

Access the dashboard at `http://localhost:3000/dashboard.html` to view the database of all detected visitors.    Age: 8-12 years

    

## 🛠️ Technologies<img src="Example/Detecting age and gender girl2.png">



- **Backend**: FastAPI (Python)    >python detect.py --image kid1.jpg

- **Computer Vision**: OpenCV (DNN Module)    Gender: Male

- **Frontend**: HTML5, CSS3, JavaScript (Canvas API)    Age: 4-6 years    

- **Models**: ONNX, Caffe    

<img src="Example/Detecting age and gender kid1.png">

## 📝 License

    >python detect.py --image kid2.jpg

This project is licensed under the MIT License.    Gender: Female

    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">
              
