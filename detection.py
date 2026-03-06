import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import joblib
import time
import pandas as pd
import os

print("Starting AI Classroom Monitoring System...")

# -------------------------
# Load YOLO
# -------------------------
model = YOLO("yolov8n.pt")

# -------------------------
# MediaPipe
# -------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1)

# -------------------------
# Load XGBoost
# -------------------------
MODEL_PATH = "xgboost_model12.pkl"

if not os.path.exists(MODEL_PATH):
    print("ERROR: xgboost_model12.pkl missing")
    exit()

loaded_model = joblib.load(MODEL_PATH)

# -------------------------
# Camera
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera not detected")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Camera Started")

# -------------------------
# Tracking
# -------------------------
student_id_counter = 0
tracked_students = {}

# -------------------------
# Logging
# -------------------------
log_data = []
last_log_time = time.time()

frame_count = 0

# -------------------------
# Angle Function
# -------------------------
def calculate_angle(p1,p2,p3):

    a=np.array(p1)
    b=np.array(p2)
    c=np.array(p3)

    ba=a-b
    bc=c-b

    cosine=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    angle=np.arccos(cosine)

    return np.degrees(angle)

# -------------------------
# Pose Features
# -------------------------
def get_pose_angles(img):

    image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(image)

    if not results.pose_landmarks:
        return None

    lm=results.pose_landmarks.landmark

    try:

        shoulder_l=[lm[11].x,lm[11].y]
        shoulder_r=[lm[12].x,lm[12].y]

        elbow_l=[lm[13].x,lm[13].y]
        elbow_r=[lm[14].x,lm[14].y]

        wrist_l=[lm[15].x,lm[15].y]
        wrist_r=[lm[16].x,lm[16].y]

        hip_l=[lm[23].x,lm[23].y]
        hip_r=[lm[24].x,lm[24].y]

        features=[

            calculate_angle(shoulder_l,elbow_l,wrist_l),
            calculate_angle(shoulder_r,elbow_r,wrist_r),

            calculate_angle(hip_l,shoulder_l,elbow_l),
            calculate_angle(hip_r,shoulder_r,elbow_r),

            shoulder_l[0],shoulder_l[1],
            shoulder_r[0],shoulder_r[1],

            elbow_l[0],elbow_l[1],
            elbow_r[0],elbow_r[1],

            wrist_l[0],wrist_l[1],
            wrist_r[0],wrist_r[1]

        ]

        return np.array(features)

    except:
        return None


# -------------------------
# Head Direction
# -------------------------
def get_head_direction(face):

    nose=face.landmark[1].x
    left_cheek=face.landmark[234].x
    right_cheek=face.landmark[454].x

    if nose < left_cheek:
        return "left"

    elif nose > right_cheek:
        return "right"

    else:
        return "center"


# -------------------------
# Attention
# -------------------------
def compute_attention(phone,down,talking):

    score=100

    if phone:
        score-=70

    if down:
        score-=40

    if talking:
        score-=30

    score=max(score,0)

    if phone:
        state="Using Phone"

    elif talking:
        state="Talking"

    elif down:
        state="Looking Down"

    elif score>80:
        state="Attentive"

    else:
        state="Distracted"

    return score,state


# -------------------------
# MAIN LOOP
# -------------------------
while True:

    ret,frame=cap.read()

    if not ret:
        break

    frame_count+=1

    # FPS BOOST
    if frame_count%5!=0:
        continue

    frame=cv2.resize(frame,(480,360))

    results=model(frame,imgsz=320,verbose=False)
    result=results[0]

    phone_boxes=[]
    student_boxes=[]
    head_dirs={}

    # -------------------------
    # Detect phones
    # -------------------------
    for box in result.boxes:

        cls=int(box.cls[0])
        label=result.names[cls]

        x1,y1,x2,y2=map(int,box.xyxy[0])

        if label=="cell phone":

            phone_boxes.append((x1,y1,x2,y2))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"Phone",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        if label=="person":

            student_boxes.append((x1,y1,x2,y2))

    # -------------------------
    # Process Students
    # -------------------------
    for (x1,y1,x2,y2) in student_boxes:

        cropped=frame[y1:y2,x1:x2]

        if cropped.size==0:
            continue

        features=get_pose_angles(cropped)

        if features is None:
            continue

        features=features[:10]
        features=features.reshape(1,-1)

        prediction=loaded_model.predict(features)[0]

        rgb=cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)
        face_results=face_mesh.process(rgb)

        looking_down=False
        head_dir="center"

        if face_results.multi_face_landmarks:

            face=face_results.multi_face_landmarks[0]

            head_dir=get_head_direction(face)

            nose=face.landmark[1].y
            chin=face.landmark[152].y

            if nose>chin-0.1:
                looking_down=True

        # phone detection
        phone_detected=False

        for px1,py1,px2,py2 in phone_boxes:

            if (px1>x1 and px2<x2 and py1>y1 and py2<y2):
                phone_detected=True

        center=((x1+x2)//2,(y1+y2)//2)

        student_id=None

        for sid,prev in tracked_students.items():

            px,py=prev

            if abs(center[0]-px)<50 and abs(center[1]-py)<50:

                student_id=sid
                tracked_students[sid]=center
                break

        if student_id is None:

            student_id_counter+=1
            student_id=student_id_counter
            tracked_students[student_id]=center

        head_dirs[student_id]=head_dir

        talking=False

        # Talking detection
        for other_id,dir2 in head_dirs.items():

            if other_id==student_id:
                continue

            if (head_dir=="left" and dir2=="right") or (head_dir=="right" and dir2=="left"):
                talking=True

        score,state=compute_attention(phone_detected,
                                      looking_down,
                                      talking)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,
                    f"ID:{student_id} {state} {score}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)

        log_data.append({
            "time":time.strftime("%H:%M:%S"),
            "student_id":student_id,
            "state":state,
            "score":score
        })

    if time.time()-last_log_time>1:

        df=pd.DataFrame(log_data)
        df.to_excel("attention_log.xlsx",index=False)

        last_log_time=time.time()

    cv2.imshow("AI Classroom Monitoring",frame)

    if cv2.waitKey(1)&0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()