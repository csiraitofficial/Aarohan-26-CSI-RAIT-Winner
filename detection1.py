


import cv2
import numpy as np
import mediapipe as mp
import math
import time
from ultralytics import YOLO


#  USER CONFIG

VIDEO_SOURCE     = 0           
FRAME_W          = 854
FRAME_H          = 480
YOLO_IMGSZ       = 320
YOLO_CONF        = 0.38
SKIP_FRAMES      = 2           
MAX_TRACK_DIST   = 130         
MAX_MISSING      = 25         

# Head-direction thresholds
DISTRACTED_THRESH = 0.10      
LOOKDOWN_THRESH   = 0.06       

# Talking detection
TALKING_DIST_PX  = 220         


#  COLOURS  (BGR)

C_GREEN  = (50,  200,  50)
C_RED    = (30,   30, 220)
C_ORANGE = (30,  140, 255)
C_BLUE   = (220, 120,  30)
C_PURPLE = (200,  60, 180)
C_WHITE  = (255, 255, 255)
C_DARK   = (20,   20,  20)
C_PANEL  = (25,   25,  25)
C_CYAN   = (200, 220,  50)

BEHAVIOR_COLORS = {
    "Attentive"   : C_GREEN,
    "Distracted"  : C_ORANGE,
    "Looking Down": C_BLUE,
    "Talking"     : C_PURPLE,
    "Using Phone" : C_RED,
}

#  MEDIAPIPE FACE MESH  (landmarks used only for maths – not drawn)

_mp_face   = mp.solutions.face_mesh
_face_mesh = _mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

LM_NOSE  = 1
LM_L_EYE = 33
LM_R_EYE = 263


def analyse_face(frame: np.ndarray, box: tuple):
    """
    Returns (head_direction, looking_down).
    head_direction in {"Center", "Looking Left", "Looking Right"}
    Never raises; returns ("Center", False) if face not found.
    """
    x1, y1, x2, y2 = box
    fh, fw = frame.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(fw, x2), min(fh, y2)

    if x2c <= x1c + 10 or y2c <= y1c + 10:
        return "Center", False

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return "Center", False

    # Upscale tiny crops so FaceMesh has enough detail
    ch, cw = crop.shape[:2]
    if cw < 80 or ch < 80:
        scale = max(80 / cw, 80 / ch)
        crop  = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                           interpolation=cv2.INTER_LINEAR)

    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = _face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return "Center", False

    lm   = result.multi_face_landmarks[0].landmark
    H, W = crop.shape[:2]

    def pt(idx):
        p = lm[idx]
        return p.x * W, p.y * H

    nose  = pt(LM_NOSE)
    l_eye = pt(LM_L_EYE)
    r_eye = pt(LM_R_EYE)

    # ── Yaw (left / right turn) 
    eye_mid_x = (l_eye[0] + r_eye[0]) / 2.0
    eye_span  = max(abs(r_eye[0] - l_eye[0]), 1.0)
    yaw       = (nose[0] - eye_mid_x) / eye_span

    if   yaw < -DISTRACTED_THRESH:
        h_dir = "Looking Right"   # nose drifts left → head turned right
    elif yaw >  DISTRACTED_THRESH:
        h_dir = "Looking Left"
    else:
        h_dir = "Center"

    # ── Pitch (looking down) 
    eye_mid_y   = (l_eye[1] + r_eye[1]) / 2.0
    face_height = max(y2c - y1c, 1)
    pitch       = (nose[1] - eye_mid_y) / face_height
    looking_dn  = pitch > LOOKDOWN_THRESH

    return h_dir, looking_dn



#  STUDENT TRACK  – weighted-average box smoothing

class StudentTrack:
    _next_id = 1

    def __init__(self, raw_box):
        self.id        = StudentTrack._next_id
        StudentTrack._next_id += 1
        b              = np.array(raw_box, dtype=float)
        self._hist     = [b.copy()] * 5   # ring buffer of last 5 raw boxes
        self._smooth   = b.copy()
        self.missing   = 0
        self.head_dir  = "Center"
        self.behavior  = "Attentive"
        self.score     = 100
        self.has_phone = False

    def update(self, raw_box):
        b = np.array(raw_box, dtype=float)
        self._hist.append(b)
        if len(self._hist) > 5:
            self._hist.pop(0)
        # recency-weighted average: weights [1,2,3,4,5]
        n = len(self._hist)
        w = np.arange(1, n + 1, dtype=float)
        w /= w.sum()
        self._smooth = sum(wi * bx for wi, bx in zip(w, self._hist))
        self.missing = 0

    def centre(self):
        x1, y1, x2, y2 = self._smooth
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def ibox(self):
        return tuple(self._smooth.astype(int))



#  TRACKING

def update_tracks(tracks: list, detections: list) -> list:
    used = [False] * len(detections)

    for t in tracks:
        tx, ty   = t.centre()
        best_i   = None
        best_d   = MAX_TRACK_DIST
        for i, (x1, y1, x2, y2) in enumerate(detections):
            if used[i]:
                continue
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = math.hypot(tx - cx, ty - cy)
            if d < best_d:
                best_d, best_i = d, i
        if best_i is not None:
            t.update(detections[best_i])
            used[best_i] = True
        else:
            t.missing += 1

    for i, box in enumerate(detections):
        if not used[i]:
            tracks.append(StudentTrack(box))

    return [t for t in tracks if t.missing < MAX_MISSING]



#  PHONE 
def check_phones(tracks: list, phone_boxes: list):
    for t in tracks:
        t.has_phone = False
        tx1, ty1, tx2, ty2 = t.ibox()
        for (px1, py1, px2, py2) in phone_boxes:
            if min(tx2, px2) > max(tx1, px1) and min(ty2, py2) > max(ty1, py1):
                t.has_phone = True
                break

#  BEHAVIOUR CLASSIFICATION

def classify_all(tracks: list):
    """
    Talking  : two students are close AND face each other
               (one "Looking Left" + one "Looking Right")
    Distracted: head turned left or right (not phone/talking/down)
    Priority  : Phone > Talking > Looking Down > Distracted > Attentive
    """
    # detect talking pairs
    talking_ids: set = set()
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a, b = tracks[i], tracks[j]
            if math.hypot(*(np.subtract(a.centre(), b.centre()))) > TALKING_DIST_PX:
                continue
            facing = (
                (a.head_dir == "Looking Left"  and b.head_dir == "Looking Right") or
                (a.head_dir == "Looking Right" and b.head_dir == "Looking Left")
            )
            if facing:
                talking_ids.add(a.id)
                talking_ids.add(b.id)

    for t in tracks:
        if t.has_phone:
            t.behavior = "Using Phone"
            t.score    = 50
        elif t.id in talking_ids:
            t.behavior = "Talking"
            t.score    = 70
        elif t.head_dir == "Looking Down":
            t.behavior = "Looking Down"
            t.score    = 80
        elif t.head_dir in ("Looking Left", "Looking Right"):
            t.behavior = "Distracted"
            t.score    = 75
        else:
            t.behavior = "Attentive"
            t.score    = 100



#  DRAWING HELPERS

_FONT  = cv2.FONT_HERSHEY_SIMPLEX


def put_label(img, text, x, y, color, fs=0.46, th=1):
    """Text with dark background pill."""
    (tw, th_), bl = cv2.getTextSize(text, _FONT, fs, th)
    p = 3
    cv2.rectangle(img, (x - p, y - th_ - p), (x + tw + p, y + bl + p), C_DARK, -1)
    cv2.putText(img, text, (x, y), _FONT, fs, color, th, cv2.LINE_AA)


def draw_student(img, t: StudentTrack):
    x1, y1, x2, y2 = t.ibox()
    color = BEHAVIOR_COLORS.get(t.behavior, C_GREEN)

    # Glow layer
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 3)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    # Main box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Corner accent lines
    L = min(16, (x2 - x1) // 5, (y2 - y1) // 5)
    for cx, cy, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + sx * L, cy),         color, 3)
        cv2.line(img, (cx, cy), (cx,          cy + sy * L), color, 3)

    # Labels (above box)
    label1 = f"ID:{t.id}  {t.behavior}"
    label2 = f"Score:{t.score}"
    put_label(img, label1, x1, y1 - 20, color, fs=0.46)
    put_label(img, label2, x1, y1 -  4, C_WHITE, fs=0.40)


def draw_panel(img, tracks: list, fps: float):
    cnt = {b: 0 for b in
           ["Attentive", "Distracted", "Looking Down", "Talking", "Using Phone"]}
    for t in tracks:
        if t.behavior in cnt:
            cnt[t.behavior] += 1

    rows = [
        ("CLASSROOM MONITOR",             C_CYAN),
        (f"Total Students  : {len(tracks)}",       C_WHITE),
        (f"Attentive        : {cnt['Attentive']}",  C_GREEN),
        (f"Distracted       : {cnt['Distracted']}", C_ORANGE),
        (f"Looking Down     : {cnt['Looking Down']}",C_BLUE),
        (f"Talking          : {cnt['Talking']}",    C_PURPLE),
        (f"Using Phone      : {cnt['Using Phone']}", C_RED),
        (f"FPS              : {fps:.1f}",           C_CYAN),
    ]

    lh, pw, px, py = 22, 230, 8, 8
    ph = len(rows) * lh + 14

    ov = img.copy()
    cv2.rectangle(ov, (px - 4, py - 4), (px + pw, py + ph), C_PANEL, -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)

    for i, (text, color) in enumerate(rows):
        bold = 2 if i == 0 else 1
        cv2.putText(img, text, (px, py + (i + 1) * lh),
                    _FONT, 0.50, color, bold, cv2.LINE_AA)

#  MAIn
def main():
    print("=" * 60)
    print("  AI Classroom Monitor  v2.0  –  loading YOLOv8n …")
    model = YOLO("yolov8n.pt")
    _ = model(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
              imgsz=YOLO_IMGSZ, verbose=False)
    print("  Model ready.  Q / Esc to quit.")
    print("=" * 60)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_SOURCE}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    WIN = "AI Classroom Monitor  [Q = quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    tracks:          list[StudentTrack] = []
    cached_persons:  list = []
    cached_phones:   list = []
    frame_idx        = 0
    fps_smooth       = 15.0
    t_prev           = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop for video files
            ret, frame = cap.read()
            if not ret:
                break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame_idx += 1

        # ── YOLO detection
        if frame_idx % SKIP_FRAMES == 0:
            results = model(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                classes=[0, 67],
                verbose=False,
            )[0]

            cached_persons.clear()
            cached_phones.clear()

            if results.boxes is not None:
                for box in results.boxes:
                    cls  = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    (cached_persons if cls == 0 else cached_phones).append(tuple(xyxy))

        # ── Tracking
        tracks = update_tracks(tracks, cached_persons)

        # ── Phone overlap 
        check_phones(tracks, cached_phones)

        # ── Head direction 
        if frame_idx % SKIP_FRAMES == 0:
            for t in tracks:
                h_dir, dn = analyse_face(frame, t.ibox())
                t.head_dir = "Looking Down" if dn else h_dir

        # ── Behaviour 
        classify_all(tracks)

        # ── Render
        for t in tracks:
            draw_student(frame, t)

        now        = time.perf_counter()
        fps_smooth = 0.85 * fps_smooth + 0.15 / max(now - t_prev, 1e-6)
        t_prev     = now

        draw_panel(frame, tracks, fps_smooth)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    _face_mesh.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
