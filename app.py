import os
import io
import csv
import time
import base64
import pickle
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify, send_file, Response
import pymysql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# --- INITIALIZATION ---
FOLDERS = [
    "TrainingImage",
    "TrainingImageLabel",
    "StudentDetails",
    "Attendance",
    "Attendance/Manually Attendance"
]

for folder in FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

DETAILS_FILE = os.path.join(BASE_DIR, "StudentDetails/StudentDetails.csv")
if not os.path.isfile(DETAILS_FILE):
    with open(DETAILS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])

CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# In-memory session tracking for stable face recognition
recognition_counters = {}

def get_cascade_detector():
    return cv2.CascadeClassifier(CASCADE_PATH)

def get_recognizer():
    if not hasattr(cv2, 'face'):
        raise Exception("OpenCV face module not found. Make sure opencv-contrib-python-headless is installed.")
    return cv2.face.LBPHFaceRecognizer_create()

def decode_base64_image(base64_string):
    """Utility to convert base64 data URI to OpenCV BGR image"""
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# --- ROUTES ---

@app.route('/')
def index():
    """Render main application dashboard UI"""
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Return dashboard summary statistics"""
    try:
        # Total registered students
        total_students = 0
        if os.path.exists(DETAILS_FILE):
            df_students = pd.read_csv(DETAILS_FILE)
            total_students = len(df_students)

        # Today's present count
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        today_file = os.path.join(BASE_DIR, f"Attendance/Attendance_{today_str}.csv")
        today_present = 0
        if os.path.exists(today_file):
            df_att = pd.read_csv(today_file)
            today_present = len(df_att['Enrollment'].unique())

        # Total trained images
        train_path = os.path.join(BASE_DIR, "TrainingImage")
        trained_images_count = len([f for f in os.listdir(train_path) if not f.startswith('.')]) if os.path.exists(train_path) else 0

        # Model status
        model_exists = os.path.exists(os.path.join(BASE_DIR, "TrainingImageLabel/Trainner.yml"))

        return jsonify({
            'success': True,
            'total_students': total_students,
            'today_present': today_present,
            'trained_images_count': trained_images_count,
            'model_ready': model_exists,
            'date': today_str
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get list of registered students"""
    try:
        if not os.path.exists(DETAILS_FILE):
            return jsonify({'success': True, 'students': []})

        df = pd.read_csv(DETAILS_FILE)
        students = df.to_dict(orient='records')
        return jsonify({'success': True, 'students': students})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_student():
    """Register student and process captured face images"""
    try:
        data = request.json
        enrollment = str(data.get('enrollment', '')).strip()
        name = str(data.get('name', '')).strip()
        images = data.get('images', [])

        if not enrollment or not name:
            return jsonify({'success': False, 'error': 'Enrollment and Name are required'}), 400

        if not images:
            return jsonify({'success': False, 'error': 'No face images provided'}), 400

        # Check duplicate enrollment in details file
        if os.path.exists(DETAILS_FILE):
            df = pd.read_csv(DETAILS_FILE)
            if enrollment in df['Enrollment'].astype(str).values:
                return jsonify({'success': False, 'error': f'Enrollment {enrollment} already exists'}), 400

        detector = get_cascade_detector()
        saved_count = 0

        for idx, base64_img in enumerate(images):
            try:
                img = decode_base64_image(base64_img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

                for (x, y, w, h) in faces:
                    face_crop = gray[y:y+h, x:x+w]
                    face_crop = cv2.resize(face_crop, (200, 200))
                    
                    saved_count += 1
                    file_path = os.path.join(BASE_DIR, f"TrainingImage/{enrollment}.{name}.{saved_count}.jpg")
                    cv2.imwrite(file_path, face_crop)
                    break # Take one face per frame image
            except Exception:
                continue

        if saved_count == 0:
            return jsonify({'success': False, 'error': 'No valid faces detected in the provided images. Please ensure your face is clearly visible.'}), 400

        # Save to StudentDetails.csv
        ts = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        with open(DETAILS_FILE, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name, date_str, time_str])

        return jsonify({
            'success': True,
            'message': f'Successfully registered {name} ({enrollment}) with {saved_count} face samples!',
            'saved_count': saved_count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train LBPH Face Recognizer model on saved images"""
    try:
        detector = get_cascade_detector()
        recognizer = get_recognizer()

        train_path = os.path.join(BASE_DIR, "TrainingImage")
        if not os.path.exists(train_path):
            return jsonify({'success': False, 'error': 'TrainingImage folder does not exist'}), 400

        image_files = [f for f in os.listdir(train_path) if not f.startswith('.')]
        if len(image_files) == 0:
            return jsonify({'success': False, 'error': 'No face images found in TrainingImage folder. Register students first.'}), 400

        face_samples = []
        ids = []
        label_map = {}
        reverse_label_map = {}

        for filename in image_files:
            image_path = os.path.join(train_path, filename)
            try:
                pil_image = Image.open(image_path).convert('L')
                image_np = np.array(pil_image, 'uint8')

                id_val = filename.split(".")[0]

                faces = detector.detectMultiScale(image_np)
                for (x, y, w, h) in faces:
                    if id_val not in label_map:
                        new_id = len(label_map) + 1
                        label_map[id_val] = new_id
                        reverse_label_map[new_id] = id_val

                    face = image_np[y:y+h, x:x+w]
                    face = cv2.resize(face, (200, 200))
                    face_samples.append(face)
                    ids.append(label_map[id_val])
            except Exception:
                continue

        if len(face_samples) == 0:
            return jsonify({'success': False, 'error': 'Could not detect faces in training dataset'}), 400

        recognizer.train(face_samples, np.array(ids))

        yml_path = os.path.join(BASE_DIR, "TrainingImageLabel/Trainner.yml")
        pkl_path = os.path.join(BASE_DIR, "TrainingImageLabel/labels.pkl")

        recognizer.save(yml_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(reverse_label_map, f)

        return jsonify({
            'success': True,
            'message': f'Model trained successfully on {len(face_samples)} face samples across {len(label_map)} registered students!',
            'samples_trained': len(face_samples),
            'students_count': len(label_map)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize face from webcam frame and record attendance"""
    try:
        yml_path = os.path.join(BASE_DIR, "TrainingImageLabel/Trainner.yml")
        pkl_path = os.path.join(BASE_DIR, "TrainingImageLabel/labels.pkl")

        if not os.path.exists(yml_path) or not os.path.exists(pkl_path):
            return jsonify({'success': False, 'error': 'Trained model not found. Please click "Train Images" first.'}), 400

        data = request.json
        base64_img = data.get('image', '')
        if not base64_img:
            return jsonify({'success': False, 'error': 'No frame image provided'}), 400

        img = decode_base64_image(base64_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = get_cascade_detector()
        recognizer = get_recognizer()
        recognizer.read(yml_path)

        with open(pkl_path, "rb") as f:
            reverse_label_map = pickle.load(f)

        df_students = pd.read_csv(DETAILS_FILE) if os.path.exists(DETAILS_FILE) else pd.DataFrame()

        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        results = []

        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        attendance_file = os.path.join(BASE_DIR, f"Attendance/Attendance_{today_str}.csv")

        # Load existing today's attendance
        marked_set = set()
        if os.path.exists(attendance_file):
            try:
                df_att = pd.read_csv(attendance_file)
                marked_set = set(df_att['Enrollment'].astype(str).tolist())
            except Exception:
                pass

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (200, 200))

            pred_id, conf = recognizer.predict(face_crop)
            accuracy = max(0, min(100, int(100 - conf)))

            if conf <= 65:
                enrollment = reverse_label_map.get(pred_id, "Unknown")
                name = "Unknown"
                if not df_students.empty and enrollment != "Unknown":
                    name_match = df_students.loc[df_students['Enrollment'].astype(str) == str(enrollment)]['Name'].values
                    if len(name_match) > 0:
                        name = name_match[0]

                is_marked = enrollment in marked_set
                attendance_marked = False

                if enrollment != "Unknown":
                    # Counter tracking for stability
                    recognition_counters[enrollment] = recognition_counters.get(enrollment, 0) + 1

                    if recognition_counters[enrollment] >= 2 and not is_marked:
                        # Record attendance
                        ts = time.time()
                        time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                        new_row = [enrollment, name, today_str, time_str]
                        if os.path.exists(attendance_file):
                            with open(attendance_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(new_row)
                        else:
                            with open(attendance_file, 'w', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])
                                writer.writerow(new_row)

                        marked_set.add(str(enrollment))
                        is_marked = True
                        attendance_marked = True

                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'enrollment': str(enrollment),
                    'name': name,
                    'confidence': float(conf),
                    'accuracy': accuracy,
                    'is_known': True,
                    'is_marked': is_marked,
                    'just_marked': attendance_marked
                })
            else:
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'enrollment': 'Unknown',
                    'name': 'Unknown Face',
                    'confidence': float(conf),
                    'accuracy': accuracy,
                    'is_known': False,
                    'is_marked': False,
                    'just_marked': False
                })

        return jsonify({
            'success': True,
            'faces': results,
            'total_detected': len(faces)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/manual-attendance', methods=['POST'])
def manual_attendance():
    """Manually add attendance entry"""
    try:
        data = request.json
        enrollment = str(data.get('enrollment', '')).strip()
        name = str(data.get('name', '')).strip()

        if not enrollment or not name:
            return jsonify({'success': False, 'error': 'Enrollment and Name are required'}), 400

        ts = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        time_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        file_name = os.path.join(BASE_DIR, f"Attendance/Attendance_{date_str}.csv")

        # Create file if not exists
        if not os.path.exists(file_name):
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])

        # Duplicate check
        df = pd.read_csv(file_name)
        if enrollment in df['Enrollment'].astype(str).values:
            return jsonify({'success': False, 'error': f'Attendance for {enrollment} already marked today'}), 400

        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name, date_str, time_str])

        return jsonify({'success': True, 'message': f'Manual attendance marked for {name} ({enrollment})'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    """Fetch attendance records for given date"""
    try:
        date_param = request.args.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
        file_name = os.path.join(BASE_DIR, f"Attendance/Attendance_{date_param}.csv")

        if not os.path.exists(file_name):
            return jsonify({'success': True, 'date': date_param, 'records': []})

        df = pd.read_csv(file_name)
        records = df.to_dict(orient='records')
        return jsonify({'success': True, 'date': date_param, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-attendance', methods=['GET'])
def export_attendance():
    """Download attendance CSV for given date"""
    try:
        date_param = request.args.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))
        file_name = os.path.join(BASE_DIR, f"Attendance/Attendance_{date_param}.csv")

        if not os.path.exists(file_name):
            # Create empty CSV if not existing
            df = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])
            df.to_csv(file_name, index=False)

        return send_file(
            file_name,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"Attendance_{date_param}.csv"
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
