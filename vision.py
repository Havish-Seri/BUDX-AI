import cv2
import mediapipe as mp
import serial
import time
import math
import webbrowser
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app)

# --- ARDUINO CONNECTION ---
try:
    ser = serial.Serial('COM7', 115200, timeout=1)
    time.sleep(2)
    print("System Online: Arduino Connected on COM7")
except Exception as e:
    ser = None
    print(f"Arduino NOT found: {e}")

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def map_val(x, in_min, in_max, out_min, out_max):
    v = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    return max(min(v, 600), 150)

# NEW: Chat Message Handler
@socketio.on('chat_message')
def handle_chat(data):
    user_text = data.get('message', '').lower()
    
    # Simple Logic (Replace this with an LLM API call if desired)
    if "status" in user_text:
        response = "All systems nominal. Arduino: " + ("Connected" if ser else "Disconnected")
    elif "reset" in user_text:
        response = "Resetting robotic arm coordinates..."
    else:
        response = f"Acknowledged: '{user_text}'. Processing command."
    
    socketio.emit('chat_response', {'response': response})

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                twist = hand_lms.landmark[17].y - hand_lms.landmark[2].y
                base_pwm = map_val(twist, -0.1, 0.1, 150, 600)
                hand_x = hand_lms.landmark[9].x
                arm_pwm = map_val(hand_x, 0.1, 0.9, 150, 600)
                t, i = hand_lms.landmark[4], hand_lms.landmark[8]
                dist = math.sqrt((t.x - i.x)**2 + (t.y - i.y)**2)
                grip_pwm = map_val(dist, 0.04, 0.2, 150, 600)

                socketio.emit('hand_data', {'x': hand_x, 'y': hand_lms.landmark[9].y})
                if ser:
                    ser.write(f'S{base_pwm}W{arm_pwm}G{grip_pwm}\n'.encode())
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    socketio.run(app, port=5000)