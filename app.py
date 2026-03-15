import cv2
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

app = Flask(__name__)

# MediaPipe model
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

typed_text = ""
gesture_count = 0
typing_start_time = None
last_press_time = 0

PRESS_DELAY = 0.35
PINCH_THRESHOLD = 40

# pointer fallback
last_ix = None
last_iy = None
lost_frames = 0
MAX_LOST = 8

keyboard = [
["Q","W","E","R","T","Y","U","I","O","P"],
["A","S","D","F","G","H","J","K","L"],
["Z","X","C","V","B","N","M"]
]

# ------------------------
# WPM
# ------------------------
def calculate_wpm(text,start):

    if start is None or len(text) < 5:
        return 0

    elapsed = time.time()-start
    minutes = elapsed/60

    return int((len(text)/5)/minutes)

# ------------------------
# Draw keyboard
# ------------------------
def draw_keyboard(frame):

    h,w,_ = frame.shape

    max_cols=max(len(r) for r in keyboard)

    key_w=int(w*0.10)
    key_h=int(h*0.13)

    start_x=int((w-max_cols*key_w)/2)
    start_y=int(h*0.05)

    key_positions=[]

    for i,row in enumerate(keyboard):

        for j,key in enumerate(row):

            x=start_x+j*key_w
            y=start_y+i*key_h

            key_positions.append((key,x,y,key_w,key_h))

            cv2.rectangle(frame,(x,y),(x+key_w,y+key_h),(255,140,0),3)

            cv2.putText(frame,
                        key,
                        (x+int(key_w*0.35),y+int(key_h*0.65)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        key_w/80,
                        (255,140,0),
                        3)

    return key_positions

# ------------------------
# Video stream
# ------------------------
def generate_frames():

    global typed_text,gesture_count,typing_start_time,last_press_time
    global last_ix,last_iy,lost_frames

    while True:

        success,frame=cap.read()

        if not success:
            continue

        frame=cv2.flip(frame,1)

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

            lost_frames=0

            hand=result.hand_landmarks[0]

            index=hand[8]

            ix=int(index.x*w)
            iy=int(index.y*h)

            last_ix,last_iy=ix,iy

        else:

            lost_frames+=1

            if last_ix is not None and lost_frames<MAX_LOST:

                ix,iy=last_ix,last_iy

        if ix is not None:

            cv2.circle(frame,(ix,iy),12,(0,255,0),-1)
            cv2.circle(frame,(ix,iy),22,(0,255,0),2)

        if result.hand_landmarks:

            hand=result.hand_landmarks[0]

            thumb=hand[4]

            tx=int(thumb.x*w)
            ty=int(thumb.y*h)

            distance=np.hypot(ix-tx,iy-ty)

            is_pinching=distance<PINCH_THRESHOLD

            index_up=hand[8].y < hand[6].y
            middle_up=hand[12].y < hand[10].y
            ring_up=hand[16].y < hand[14].y

            if is_pinching:

                for key,x,y,kw,kh in key_positions:

                    if x<ix<x+kw and y<iy<y+kh:

                        cv2.rectangle(frame,(x,y),(x+kw,y+kh),(0,255,255),3)

                        if time.time()-last_press_time>PRESS_DELAY:

                            typed_text+=key
                            gesture_count+=1

                            if typing_start_time is None:
                                typing_start_time=time.time()

                            last_press_time=time.time()

            else:

                if index_up and middle_up and not ring_up:

                    if time.time()-last_press_time>0.6:

                        typed_text+=" "
                        gesture_count+=1
                        last_press_time=time.time()

                if index_up and middle_up and ring_up:

                    if typed_text and time.time()-last_press_time>0.6:

                        typed_text=typed_text[:-1]
                        gesture_count+=1
                        last_press_time=time.time()

        ret,buffer=cv2.imencode('.jpg',frame)

        frame=buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

# ------------------------
# Routes
# ------------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():

    wpm=calculate_wpm(typed_text,typing_start_time)

    elapsed=0

    if typing_start_time:
        elapsed=int(time.time()-typing_start_time)

    return jsonify({
        "text":typed_text,
        "wpm":wpm,
        "chars":len(typed_text),
        "gestures":gesture_count,
        "time":elapsed
    })

if __name__=="__main__":
    app.run(debug=True)