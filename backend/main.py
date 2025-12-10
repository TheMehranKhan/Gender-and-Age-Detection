from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from PIL import Image

app = FastAPI(title="Gender and Age Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import json
import os
from datetime import datetime
from PIL import Image

# Try to import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("DeepFace is available.")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("DeepFace is NOT available. Install with 'pip install deepface tf-keras'")

app = FastAPI(title="Gender and Age Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
# Old Caffe models for Age/Gender
ageProto = "/workspaces/Gender-and-Age-Detection/age_deploy.prototxt"
ageModel = "/workspaces/Gender-and-Age-Detection/age_net.caffemodel"
genderProto = "/workspaces/Gender-and-Age-Detection/gender_deploy.prototxt"
genderModel = "/workspaces/Gender-and-Age-Detection/gender_net.caffemodel"

# New ONNX models for Face Detection and Recognition
yunet_path = "/workspaces/Gender-and-Age-Detection/face_detection_yunet_2023mar.onnx"
sface_path = "/workspaces/Gender-and-Age-Detection/face_recognition_sface_2021dec.onnx"
yolox_path = "/workspaces/Gender-and-Age-Detection/yolox_s.onnx"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Initialize models
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize YuNet
face_detector = cv2.FaceDetectorYN.create(
    model=yunet_path,
    config="",
    input_size=(320, 320),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Initialize SFace
face_recognizer = cv2.FaceRecognizerSF.create(
    model=sface_path,
    config="",
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Initialize YOLOX
# YOLOX classes (COCO 80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# YOLOX Preprocessing and Postprocessing
def yolox_preprocess(img, input_size=(640, 640)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    return padded_img, r

def yolox_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((1, shape[1], 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs

yolox_net = cv2.dnn.readNetFromONNX(yolox_path)

# Simple Database
DB_FILE = "people_db.json"
people_db = {}

def load_db():
    global people_db
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays for embeddings
                for pid, pdata in data.items():
                    pdata['embedding'] = np.array(pdata['embedding'], dtype=np.float32)
                    # Initialize path if not exists
                    if 'path' not in pdata:
                        pdata['path'] = []
                people_db = data
        except Exception as e:
            print(f"Error loading DB: {e}")
            people_db = {}

def save_db():
    # Convert numpy arrays to lists for JSON serialization
    serializable_db = {}
    for pid, pdata in people_db.items():
        serializable_db[pid] = pdata.copy()
        serializable_db[pid]['embedding'] = pdata['embedding'].tolist()
        # Limit path history to last 20 points to save space
        if 'path' in serializable_db[pid]:
             serializable_db[pid]['path'] = serializable_db[pid]['path'][-20:]
    
    with open(DB_FILE, 'w') as f:
        json.dump(serializable_db, f)

load_db()

def get_person_id(embedding):
    best_match_id = None
    max_score = 0.0
    # Cosine similarity threshold for SFace is typically around 0.363
    threshold = 0.363

    for pid, pdata in people_db.items():
        db_embedding = pdata['embedding']
        score = face_recognizer.match(embedding, db_embedding, cv2.FaceRecognizerSF_FR_COSINE)
        if score > max_score:
            max_score = score
            best_match_id = pid
    
    if max_score > threshold:
        return best_match_id, max_score
    return None, max_score

@app.post("/detect")
async def detect_gender_age(file: UploadFile = File(...), enable_deepface: bool = False):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    height, width, _ = img.shape
    
    # --- Object Detection (YOLOX) ---
    input_shape = (640, 640)
    preprocessed_img, ratio = yolox_preprocess(img, input_shape)
    blob = cv2.dnn.blobFromImage(preprocessed_img)
    yolox_net.setInput(blob)
    outputs = yolox_net.forward()
    outputs = yolox_postprocess(outputs, input_shape)[0]

    boxes = outputs[:, :4]
    scores = outputs[:, 4:5] * outputs[:, 5:]
    
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    
    dets = cv2.dnn.NMSBoxes(boxes_xyxy, scores.max(1), 0.5, 0.5)
    
    detected_objects = []
    if len(dets) > 0:
        for i in dets.flatten():
            class_id = np.argmax(scores[i])
            score = scores[i][class_id]
            if score < 0.5: continue
            
            # Skip 'person' class in object detection if we want to rely on Face Detection for people
            # But user asked for object detection, so let's include everything except maybe person if it overlaps?
            # Let's include everything for now.
            
            box = boxes_xyxy[i]
            detected_objects.append({
                "class": COCO_CLASSES[class_id],
                "score": float(score),
                "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            })

    # --- Face Detection & Recognition ---
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(img)
    
    results = []
    if faces is not None:
        for face in faces:
            # YuNet returns [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            box = face[0:4].astype(int)
            x, y, w, h = box
            
            # Align and crop face for recognition
            aligned_face = face_recognizer.alignCrop(img, face)
            
            # Get embedding
            embedding = face_recognizer.feature(aligned_face)
            
            # Identify person
            person_id, score = get_person_id(embedding)
            
            is_new = False
            if person_id is None:
                # New person
                person_id = f"Person_{len(people_db) + 1}"
                people_db[person_id] = {
                    "embedding": embedding,
                    "first_seen": datetime.now().isoformat(),
                    "visits": 1,
                    "last_seen": datetime.now().isoformat(),
                    "path": []
                }
                is_new = True
            else:
                # Update existing person
                people_db[person_id]["visits"] += 1
                people_db[person_id]["last_seen"] = datetime.now().isoformat()
            
            # Update Path
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            if 'path' not in people_db[person_id]:
                people_db[person_id]['path'] = []
            people_db[person_id]['path'].append([center_x, center_y])
            
            # Age and Gender Detection
            # Optimization: Only run if not already in DB or if it's a new person
            # We can also update it periodically, but for now let's cache it.
            
            if 'gender' in people_db[person_id] and 'age' in people_db[person_id]:
                gender = people_db[person_id]['gender']
                age = people_db[person_id]['age']
            else:
                # Add padding for better age/gender detection context
                padding = 20
                face_img = img[max(0, y-padding):min(y+h+padding, height),
                            max(0, x-padding):min(x+w+padding, width)]
                
                if face_img.size == 0:
                    gender = "Unknown"
                    age = "Unknown"
                else:
                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                
                # Store in DB
                people_db[person_id]['gender'] = gender
                people_db[person_id]['age'] = age

            # --- DeepFace Analysis (Emotion & Race) ---
            emotion = people_db[person_id].get('emotion', 'Unknown')
            race = people_db[person_id].get('race', 'Unknown')

            if DEEPFACE_AVAILABLE and enable_deepface and (emotion == 'Unknown' or is_new):
                padding = 20
                face_img = img[max(0, y-padding):min(y+h+padding, height),
                            max(0, x-padding):min(x+w+padding, width)]
                
                if face_img.size > 0:
                    try:
                        # DeepFace analysis
                        analysis = DeepFace.analyze(face_img, actions=['emotion', 'race'], enforce_detection=False, silent=True)
                        if analysis:
                            emotion = analysis[0]['dominant_emotion']
                            race = analysis[0]['dominant_race']
                            people_db[person_id]['emotion'] = emotion
                            people_db[person_id]['race'] = race
                    except Exception as e:
                        print(f"DeepFace Error: {e}")

            # --- Liveness Detection (Temporal Heuristic) ---
            liveness = "Verifying..."
            path_len = len(people_db[person_id].get('path', []))
            if path_len > 15:
                liveness = "Live"
            
            results.append({
                "gender": gender,
                "age": age,
                "emotion": emotion,
                "race": race,
                "liveness": liveness,
                "face_box": [int(x), int(y), int(x+w), int(y+h)], # Format for frontend
                "person_id": person_id,
                "visits": people_db[person_id]["visits"],
                "is_new": is_new,
                "path": people_db[person_id]['path'][-20:] # Send last 20 points
            })

    if faces is not None:
        save_db()

    return {"results": results, "objects": detected_objects}

@app.get("/people")
async def get_people():
    results = []
    for pid, data in people_db.items():
        results.append({
            "id": pid,
            "visits": data.get("visits", 0),
            "first_seen": data.get("first_seen", ""),
            "last_seen": data.get("last_seen", ""),
            "gender": data.get("gender", "Unknown"),
            "age": data.get("age", "Unknown")
        })
    # Sort by last_seen descending
    results.sort(key=lambda x: x["last_seen"], reverse=True)
    return results


@app.get("/")
async def root():
    return {"message": "Gender and Age Detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)