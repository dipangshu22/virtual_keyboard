import cv2
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

app = Flask(__name__)

# ---------------------------
# MediaPipe
# ---------------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ---------------------------
# STATE
# ---------------------------
typed_text = ""
gesture_count = 0
typing_start_time = None
last_press_time = 0

PRESS_DELAY = 0.25

distance_buffer = []
BUFFER_SIZE = 5
is_pinching = False

space_buffer = []
backspace_buffer = []
GESTURE_BUFFER = 5

last_ix, last_iy = None, None

keyboard = [
["Q","W","E","R","T","Y","U","I","O","P"],
["A","S","D","F","G","H","J","K","L"],
["Z","X","C","V","B","N","M"]
]

# ---------------------------
def calculate_wpm(text,start):
    if start is None or len(text) < 5:
        return 0
    elapsed = time.time()-start
    return int((len(text)/5)/(elapsed/60))

# ---------------------------
def draw_keyboard(frame):
    h,w,_ = frame.shape

    max_cols=max(len(r) for r in keyboard)

    key_w=int(w*0.08)
    key_h=int(h*0.10)

    start_x=int((w-max_cols*key_w)/2)
    start_y=int(h*0.50)   # adjusted after camera shift

    key_positions=[]

    for i,row in enumerate(keyboard):
        for j,key in enumerate(row):

            x=start_x+j*key_w
            y=start_y+i*key_h

            key_positions.append((key,x,y,key_w,key_h))

            cv2.rectangle(frame,(x,y),(x+key_w,y+key_h),(255,140,0),2)

            cv2.putText(frame,key,
                        (x+int(key_w*0.35),y+int(key_h*0.65)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        key_w/100,
                        (255,140,0),2)

    return key_positions

# ---------------------------
def is_finger_up(tip, pip):
    return (pip.y - tip.y) > 0.04

# ---------------------------
def generate_frames():

    global typed_text,gesture_count,typing_start_time,last_press_time
    global last_ix,last_iy,distance_buffer,is_pinching
    global space_buffer,backspace_buffer

    while True:

        success,frame=cap.read()
        if not success:
            continue

        frame=cv2.flip(frame,1)

        # ---------------------------
        # CAMERA SHIFT UP (IMPORTANT)
        # ---------------------------
        h, w, _ = frame.shape
        crop_top = int(h * 0.15)
        frame = frame[crop_top:h, 0:w]
        frame = cv2.resize(frame, (w, h))

        key_positions=draw_keyboard(frame)

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        mp_image=mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result=detector.detect(mp_image)

        h,w,_=frame.shape
        ix,iy=None,None

        if result.hand_landmarks:

            hand=result.hand_landmarks[0]

            index=hand[8]
            ix=int(index.x*w)
            iy=int(index.y*h)

            # smoothing pointer
            if last_ix is not None:
                ix=int(0.7*last_ix+0.3*ix)
                iy=int(0.7*last_iy+0.3*iy)

            last_ix,last_iy=ix,iy

        if ix is not None:
            cv2.circle(frame,(ix,iy),10,(0,255,0),-1)

        # ---------------------------
        # HAND LOGIC
        # ---------------------------
        if result.hand_landmarks:

            hand=result.hand_landmarks[0]

            thumb=hand[4]
            wrist=hand[0]
            middle_base=hand[9]

            tx,ty=int(thumb.x*w),int(thumb.y*h)
            wx,wy=int(wrist.x*w),int(wrist.y*h)
            mx,my=int(middle_base.x*w),int(middle_base.y*h)

            # NORMALIZED DISTANCE
            finger_dist=np.hypot(ix-tx,iy-ty)
            hand_size=np.hypot(wx-mx,wy-my)

            if hand_size==0:
                hand_size=1

            norm_dist=finger_dist/hand_size

            # SMOOTHING
            distance_buffer.append(norm_dist)
            if len(distance_buffer)>BUFFER_SIZE:
                distance_buffer.pop(0)

            smooth_dist=sum(distance_buffer)/len(distance_buffer)

            # HYSTERESIS
            if not is_pinching and smooth_dist<0.35:
                is_pinching=True
            elif is_pinching and smooth_dist>0.45:
                is_pinching=False

            cv2.putText(frame,f"{smooth_dist:.2f}",(20,60),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

            # HOVER
            for key,x,y,kw,kh in key_positions:
                if x<ix<x+kw and y<iy<y+kh:
                    cv2.rectangle(frame,(x,y),(x+kw,y+kh),(255,255,0),2)

            # CLICK
            if is_pinching:
                for key,x,y,kw,kh in key_positions:
                    if x<ix<x+kw and y<iy<y+kh:
                        if time.time()-last_press_time>PRESS_DELAY:
                            typed_text+=key
                            gesture_count+=1
                            if typing_start_time is None:
                                typing_start_time=time.time()
                            last_press_time=time.time()

            # ---------------------------
            # GESTURES
            # ---------------------------
            index_up=is_finger_up(hand[8],hand[6])
            middle_up=is_finger_up(hand[12],hand[10])
            ring_up=is_finger_up(hand[16],hand[14])

            # SPACE
            space_detect=index_up and middle_up and not ring_up
            space_buffer.append(space_detect)
            if len(space_buffer)>GESTURE_BUFFER:
                space_buffer.pop(0)

            if sum(space_buffer)>GESTURE_BUFFER*0.7 and not is_pinching:
                if time.time()-last_press_time>0.6:
                    typed_text+=" "
                    last_press_time=time.time()

            # BACKSPACE
            back_detect=index_up and middle_up and ring_up
            backspace_buffer.append(back_detect)
            if len(backspace_buffer)>GESTURE_BUFFER:
                backspace_buffer.pop(0)

            if sum(backspace_buffer)>GESTURE_BUFFER*0.7 and not is_pinching:
                if typed_text and time.time()-last_press_time>0.6:
                    typed_text=typed_text[:-1]
                    last_press_time=time.time()

        # SHOW TEXT
        cv2.putText(frame,typed_text[-30:],(20,h-20),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

# ---------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        "text":typed_text,
        "wpm":calculate_wpm(typed_text,typing_start_time),
        "chars":len(typed_text),
        "gestures":gesture_count
    })

# ---------------------------
if __name__=="__main__":
    app.run(debug=True)