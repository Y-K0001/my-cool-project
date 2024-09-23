import os
import time
import cv2
import pandas as pd
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# Load the Excel file and clean data
excel_file_path = 'C:\\Users\\Yan13\\Desktop\\1\\HAP0824TWD.xlsx'

df = pd.read_excel(excel_file_path, header=0)

# Verify columns and rename as needed
df.columns = ['Part No.', 'English name']

# Clean the DataFrame
df_cleaned = df.dropna()

# Function to search Excel by Part No.
def get_part_details(part_no):
    part_no_str = str(part_no).strip()
    part_info = df_cleaned[df_cleaned['Part No.'].astype(str).str.strip() == part_no_str]

    if not part_info.empty:
        return part_info.iloc[0]['Part No.'], part_info.iloc[0]['English name']
    return None, None

# Path to save captured images
save_path = os.path.join('static', 'captured_images')

# Ensure the path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Load YOLO model
model = YOLO(r"best.pt")

# Function to capture image from the camera and detect objects
def capture_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot access the camera")
        return None, []

    ret, frame = cap.read()

    if ret:
        # Save the image
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(save_path, filename)

        # Detect objects with YOLO
        results = model(frame, conf=0.3)
        detected_objects = []

        # Debugging: print detection results
        print(f"YOLO Detection Results: {results}")  # Print the entire results object

        for result in results:
            print(f"Result: {result}")  # Print each detection result

            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = box.conf.item()

                # Debugging: print each detected object and its confidence
                print(f"Detected object: {label} with confidence: {confidence}")

                # Check Part No. and look up in Excel
                part_no, english_name = get_part_details(label)

                if part_no and english_name:
                    label_text = f'Part No: {part_no} - {english_name} ({round(confidence, 2)})'
                    detected_objects.append({
                        'part_no': part_no,
                        'name': english_name,
                        'confidence': round(confidence, 2)
                    })
                else:
                    label_text = f'{label} - Not found ({round(confidence, 2)})'
                    detected_objects.append({
                        'part_no': label,
                        'name': 'Not found',
                        'confidence': round(confidence, 2)
                    })

                # Draw bounding boxes on the image
                if hasattr(box, 'xyxy'):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    print("No bounding box information found.")

        # Save the image with bounding boxes
        cv2.imwrite(filepath, frame)
        print(f"Image saved to: {filepath}")

        cap.release()
        return filename, detected_objects
    else:
        cap.release()
        print("Failed to capture image")
        return None, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data.html')
def data_page():
    return render_template('data.html') 

@app.route("/index2")
def index_page():
    return render_template("index111.html")

@app.route("/nextpage.html")
def next_page():
    return render_template("nextpage.html")

@app.route('/capture', methods=['POST'])
def capture():
    filename, detected_objects = capture_image()

    if filename:
        # Debugging: print captured filename and objects
        print(f"Captured file saved as: /static/captured_images/{filename}")
        print(f"Captured objects: {detected_objects}")
        return jsonify({'status': 'success', 'filename': filename, 'objects': detected_objects})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to capture image'})

@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)

    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

