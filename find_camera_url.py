"""
A9 WiFi Camera – Stream URL Finder
------------------------------------
1. Replace YOUR_CAMERA_IP below with your actual IP (e.g. 192.168.1.105)
2. Run:  python find_camera_url.py
3. It will tell you exactly which URL works for your camera
"""

import cv2
import time

# ─────────────────────────────────────────
#  ← PUT YOUR CAMERA IP HERE
CAMERA_IP = "192.168.1.101"
# ─────────────────────────────────────────

# All known URL formats used by A9 / HD Wifi Cam Pro cameras
URLS_TO_TRY = [
    f"rtsp://{CAMERA_IP}/live",
    f"rtsp://{CAMERA_IP}:554/live",
    f"rtsp://{CAMERA_IP}:554/stream",
    f"rtsp://{CAMERA_IP}:554/ch0_0.264",
    f"rtsp://admin:admin@{CAMERA_IP}/live",
    f"rtsp://admin:admin@{CAMERA_IP}:554/live",
    f"rtsp://admin:@{CAMERA_IP}:554/live",
    f"rtsp://{CAMERA_IP}:1935/live/stream",
    f"http://{CAMERA_IP}:8080/video",
    f"http://{CAMERA_IP}:80/video",
    f"http://{CAMERA_IP}/stream",
    f"http://{CAMERA_IP}:8080/?action=stream",
    f"http://{CAMERA_IP}/videostream.cgi",
]

print("=" * 55)
print(f"  Scanning A9 camera at IP: {CAMERA_IP}")
print("=" * 55)

working_url = None

for url in URLS_TO_TRY:
    print(f"\n  Trying: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # Give it 3 seconds to connect
    timeout = time.time() + 3
    connected = False
    while time.time() < timeout:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                connected = True
                break
        time.sleep(0.2)

    cap.release()

    if connected:
        print(f"  ✅  SUCCESS! Working URL:")
        print(f"\n      {url}\n")
        working_url = url
        break
    else:
        print(f"  ❌  No response")

print("=" * 55)
if working_url:
    print(f"\n  Copy this into classroom_monitor_v3.py line 34:")
    print(f'\n      VIDEO_SOURCE = "{working_url}"\n')
else:
    print("\n  No URL worked automatically.")
    print("  Try these manual steps:")
    print(f"  1. Open HD Wifi Cam Pro app settings")
    print(f"  2. Look for 'Device Info' or 'Network Info'")
    print(f"  3. Check if there's a different port number shown")
    print(f"  4. Try pinging: ping {CAMERA_IP}")
print("=" * 55)
