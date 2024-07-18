import sys
import os
import cv2
import numpy as np
import json
import time
from collections import deque
import pandas as pd  # Import pandas for DataFrame handling
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import inspect
from protocols.ziv_protocols import *
from classes import Chamber, OdorColumn
from functions import *

# Add the parent directory of 'protocols' to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)

# Initialize 20 rectangles with default positions and adjusted sizes
rectangles = [(50 + i * 60, 50, 200, 15) for i in range(20)]
rectangles_file = 'rectangles.json'

show_fly_detection = True
show_rectangles = True
video_width = 800
video_height = 450
fly_positions = []
frame_deque = deque(maxlen=5)
fly_positions_vector = [0] * 20
fly_positions_matrix = []
duration_to_process = 30

def save_rectangles():
    try:
        with open(rectangles_file, 'w') as file:
            json.dump(rectangles, file)
    except IOError as e:
        print(f"Error saving rectangles: {e}")

def load_rectangles():
    global rectangles
    if os.path.exists(rectangles_file):
        try:
            with open(rectangles_file, 'r') as file:
                rectangles = json.load(file)
        except IOError as e:
            print(f"Error loading rectangles: {e}")

def process_frame(frame):
    global show_fly_detection, fly_positions, fly_positions_vector
    frame = cv2.resize(frame, (video_width, video_height))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_deque.append(gray_frame)
    avg_frame = np.mean(frame_deque, axis=0).astype(np.uint8)
    frame_diff = cv2.absdiff(gray_frame, avg_frame)
    blur_diff = cv2.GaussianBlur(frame_diff, (5, 5), 0)
    _, fgMask = cv2.threshold(blur_diff, 5, 255, cv2.THRESH_BINARY)
    fly_positions = []

    if show_fly_detection:
        for idx, (x, y, w, h) in enumerate(rectangles):
            mask = np.zeros_like(fgMask)
            mask[y:y+h, x:x+w] = 1
            masked_fg = cv2.bitwise_and(fgMask, fgMask, mask=mask)
            contours, _ = cv2.findContours(masked_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 15:
                    cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                    fly_center_x = cx + cw // 2
                    fly_center_y = cy + ch // 2
                    fly_positions.append((fly_center_x, fly_center_y))
                    relative_position = int(((fly_center_x - x) / w) * 100)
                    fly_positions_vector[idx] = relative_position
                    Chamber._instances[idx].update_fly_location(relative_position)

    return frame, fgMask, blur_diff

# Function to get the list of Python files in the 'protocols' subfolder
def get_protocol_files():
    protocols_dir = os.path.join(os.path.dirname(__file__), 'protocols')
    return [f for f in os.listdir(protocols_dir) if f.endswith('.py')]

# Update options to hold the names of the files in the 'protocols' subfolder
options = get_protocol_files()

# Create 20 Chamber objects with unique IDs from 1 to 20
for id in range(1, 21):
    Chamber(id)

# Create OdorColumn objects for Column A and Column B
OdorColumn('A')
OdorColumn('B')

@app.route('/', methods=['GET', 'POST'])
def index():
    load_rectangles()
    return render_template('index.html', chambers=Chamber._instances, odor_columns=OdorColumn._instances, options=options, rectangles=rectangles, fly_positions_vector=fly_positions_vector)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_fly_detection')
def toggle_fly_detection():
    global show_fly_detection
    show_fly_detection = not show_fly_detection
    return jsonify(status="success")

@app.route('/toggle_rectangles')
def toggle_rectangles():
    global show_rectangles
    show_rectangles = not show_rectangles
    return jsonify(status="success")

@app.route('/update_rectangle', methods=['POST'])
def update_rectangle():
    global rectangles
    data = request.json
    index = data.get('index')
    x = data.get('x')
    y = data.get('y')
    w = data.get('width')
    h = data.get('height')
    if index is not None and 0 <= index < len(rectangles):
        rectangles[index] = (x, y, w, h)
        save_rectangles()
        return jsonify(status="success")
    return jsonify(status="error")

@app.route('/get_fly_positions')
def get_fly_positions():
    return jsonify(fly_positions)

@app.route('/get_fly_positions_vector')
def get_fly_positions_vector():
    return jsonify(fly_positions_vector)

@app.route('/get_functions', methods=['POST'])
def get_functions():
    selected_option = request.json['selected_option']
    module_name, _ = os.path.splitext(selected_option)  # Remove the .py extension
    module = __import__(f'protocols.{module_name}', fromlist=[None])
    functions = [func for func, obj in inspect.getmembers(module, inspect.isfunction) if obj.__module__ == module.__name__]
    return jsonify(functions=functions)

@app.route('/start_experiment', methods=['POST'])
def start_experiment():
    selected_option = request.json['experiment_option']
    function_name = request.json['function_name']
    module_name, _ = os.path.splitext(selected_option)  # Remove the .py extension
    # Import and execute the selected protocol
    module = __import__(f'protocols.{module_name}', fromlist=[None])
    if hasattr(module, function_name):
        func = getattr(module, function_name)
        func()
    print(f'Experiment {selected_option} has started with function {function_name}')
    return jsonify(message=f'Experiment {selected_option} with function {function_name} has started and completed')

@app.route('/get_chamber_states', methods=['GET'])
def get_chamber_states():
    chambers_data = [{'id': chamber.chamber_id, 'leftShock': chamber.leftShock, 'rightShock': chamber.rightShock, 'currentFlyLoc': chamber.currentFlyLoc} for chamber in Chamber._instances]
    odor_columns_data = [{'id': odor_column.column_id, 'leftOdor': odor_column.leftOdor, 'rightOdor': odor_column.rightOdor, 'airFlow': odor_column.airFlow} for odor_column in OdorColumn._instances]
    return jsonify(chambers=chambers_data, odor_columns=odor_columns_data)

@app.route('/shock_left')
def shock_left():
    Chamber.shockLeft()
    return redirect(url_for('index'))

@app.route('/remove_shock_left')
def remove_shock_left():
    Chamber.removeShockLeft()
    return redirect(url_for('index'))

@app.route('/shock_right')
def shock_right():
    Chamber.shockRight()
    return redirect(url_for('index'))

@app.route('/remove_shock_right')
def remove_shock_right():
    Chamber.removeShockRight()
    return redirect(url_for('index'))

@app.route('/activate_odor_left')
def activate_odor_left():
    OdorColumn.activateOdorLeft()
    return redirect(url_for('index'))

@app.route('/activate_odor_right')
def activate_odor_right():
    OdorColumn.activateOdorRight()
    return redirect(url_for('index'))

@app.route('/disable_odor_left')
def disable_odor_left():
    OdorColumn.disableOdorLeft()
    return redirect(url_for('index'))

@app.route('/disable_odor_right')
def disable_odor_right():
    OdorColumn.disableOdorRight()
    return redirect(url_for('index'))

@app.route('/activate_air_flow')
def activate_air_flow():
    OdorColumn.activateAirflow()
    return redirect(url_for('index'))

@app.route('/disable_air_flow')
def disable_air_flow():
    OdorColumn.disableAirflow()
    return redirect(url_for('index'))

def generate_frames():
    load_rectangles()
    cap = cv2.VideoCapture('output.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(duration_to_process * fps)
    frame_count = 0

    while frame_count < total_frames:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (video_width, video_height))
        overlay_frame, fgMask, blur_diff = process_frame(frame)
        fly_positions_matrix.append(fly_positions_vector.copy())
        ret, buffer = cv2.imencode('.jpg', overlay_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elapsed_time = time.time() - start_time
        time.sleep(max(0, (1.0 / fps) - elapsed_time / 2))
        frame_count += 1

    # Create a DataFrame from the fly positions matrix
    columns = ['Event'] + [f'Fly_{i+1}' for i in range(20)]
    data = {'Event': [''] * len(fly_positions_matrix)}
    for i in range(20):
        data[f'Fly_{i+1}'] = [row[i] for row in fly_positions_matrix]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save the DataFrame as a CSV file
    df.to_csv('outputs/fly_positions_matrix.csv', index=False)
    
    cap.release()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
