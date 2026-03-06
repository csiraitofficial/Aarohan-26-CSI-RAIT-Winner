import cv2, numpy as np, urllib.request, threading, time

url = "http://192.168.5.147:8080/video"
frame_holder = [None]

def reader():
    stream = urllib.request.urlopen(url, timeout=10)
    buf = b""
    while True:
        buf += stream.read(8192)
        s = buf.find(b'\xff\xd8')
        e = buf.find(b'\xff\xd9')
        if s != -1 and e != -1 and e > s:
            jpg = buf[s:e+2]; buf = buf[e+2:]
            f = cv2.imdecode(np.frombuffer(jpg, np.uint8), 1)
            if f is not None: frame_holder[0] = f

t = threading.Thread(target=reader, daemon=True)
t.start()
time.sleep(3)
while True:
    f = frame_holder[0]
    if f is not None:
        cv2.imshow("Live", f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()