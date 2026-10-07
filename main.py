import tkinter as tk
from tkinter import messagebox, Label, Button, Entry
import cv2
import csv
import os
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import pymysql
import pickle
recognizer_global = None
if not hasattr(cv2, 'face'):
    raise Exception("Install opencv-contrib-python (pip install opencv-contrib-python)")
# Face label mappings
label_map = {}
reverse_label_map = {}

# --- INITIALIZATION ---
for folder in ["TrainingImage", "TrainingImageLabel", "StudentDetails", "Attendance", "Attendance/Manually Attendance"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

details_file = "StudentDetails/StudentDetails.csv"
if not os.path.isfile(details_file):
    with open(details_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])

# --- DATABASE HELPER ---
def get_db_connection(db_name):
    try:
        conn = pymysql.connect(host='localhost', user='root', password='', db=db_name)
        return conn
    except Exception as e:
        messagebox.showerror("DB Error", f"Could not connect to MySQL: {e}")
        return None

# --- CORE FUNCTIONS (Moved up so UI can see them) ---

def clear():
    txt.delete(0, 'end')

def clear1():
    txt2.delete(0, 'end')

def is_face_already_registered(face_img):
    global recognizer_global

    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        return False

    if recognizer_global is None:
        recognizer_global = cv2.face.LBPHFaceRecognizer_create()
        recognizer_global.read("TrainingImageLabel/Trainner.yml")

    face_img = cv2.resize(face_img, (200, 200))
    pred_id, conf = recognizer_global.predict(face_img)

    return conf < 30

def check_duplicate_in_dataset(new_face):

    path = "TrainingImage"

    if not os.path.exists(path):
        return False

    files = os.listdir(path)

    # Only check first 300 images (speed optimization)
    files = files[-50:]

    new_face_resized = cv2.resize(new_face, (200, 200))

    for file in files:

        if file.startswith("."):
            continue

        img_path = os.path.join(path, file)

        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (200, 200))

            diff = np.mean(cv2.absdiff(img, new_face_resized))

            if diff < 18:
                return True

        except:
            continue

    return False

def take_img():
    enrollment = txt.get()
    name = txt2.get()

    # ---- VALIDATION ----
    if not enrollment or not name:
        messagebox.showwarning("Warning", "Enrollment & Name are required!")
        return

    # ---- DUPLICATE ENROLLMENT CHECK ----
    if os.path.exists(details_file):
        df = pd.read_csv(details_file)

        if enrollment in df['Enrollment'].astype(str).values:
            messagebox.showwarning("Duplicate Entry", "Enrollment already exists!")
            return

    try:
        # ---- LOAD HAARCASCADE ----
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_cascade = os.path.join(base_path, 'haarcascade_frontalface_default.xml')

        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        detector = cv2.CascadeClassifier(cascade_path)

        if detector.empty():
            messagebox.showerror("Error", "Could not load Haarcascade XML file!")
            return

        # ---- START CAMERA ----
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return
        time.sleep(1)

        count = 0

        # ---- IMAGE CAPTURE LOOP ----
        while count < 50:
            ret, img = cam.read()
            if not ret:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(100,100))

            for (x, y, w, h) in faces:

                if w < 100 or h < 100:
                    continue

                face_crop = gray[y:y+h, x:x+w]

                #  Only check duplicates AFTER 25 images (important fix)
                if count > 30:
                    if check_duplicate_in_dataset(face_crop):
                        cv2.putText(img, "Move Face Slightly", (x, y-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                        continue

                #  Blur check (relaxed)
                laplacian_var = cv2.Laplacian(face_crop, cv2.CV_64F).var()
                if laplacian_var < 15:
                    continue

                #  Resize BEFORE saving
                face_crop = cv2.resize(face_crop, (200, 200))

                #  Save image
                count += 1
                cv2.imwrite(f"TrainingImage/{enrollment}.{name}.{count}.jpg", face_crop)

                #  Draw UI
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, f"Images: {count}/50", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #  STOP EXACTLY AT 50 (correct place)
                if count >= 50:
                    break

            if count >= 50:
                break
            
            try:
                cv2.imshow('Taking Images (Press Q to Stop)', img)
                
            except:
                pass

            # ---- EXIT CONDITIONS ----
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ---- RELEASE CAMERA ----
        cam.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        # ---- SAVE STUDENT DETAILS ----
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        with open(details_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name, date, timestamp])

        # ---- SUCCESS MESSAGE ----
        Notification.configure(
            text=f"Captured {count} Images for {name}",
            bg="SpringGreen3"
        )

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        
def trainimg():
    global label_map, reverse_label_map
    label_map.clear()
    reverse_label_map.clear()
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_cascade = os.path.join(base_path, 'haarcascade_frontalface_default.xml')

        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier(cascade_path)

        path = "TrainingImage"
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]

        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')

            filename = os.path.split(imagePath)[-1]

            try:
                id_val = filename.split(".")[0]   # Enrollment number
            except:
                continue

            faces = detector.detectMultiScale(imageNp)

            for (x, y, w, h) in faces:

                if id_val not in label_map:
                    new_id = len(label_map) + 1
                    label_map[id_val] = new_id
                    reverse_label_map[new_id] = id_val

                face = imageNp[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                faceSamples.append(face)
                ids.append(label_map[id_val])

        if len(faceSamples) == 0:
            messagebox.showwarning("Warning", "No faces found in TrainingImage folder!")
            return

        recognizer.train(faceSamples, np.array(ids))
        recognizer.save("TrainingImageLabel/Trainner.yml")
        global recognizer_global
        recognizer_global = None

        with open("TrainingImageLabel/labels.pkl", "wb") as f:
            pickle.dump(reverse_label_map, f)

       
        Notification.configure(text="Model Trained Successfully", bg="olive drab")

    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")

def automatic_attendance():
    global reverse_label_map
    if not os.path.exists("TrainingImageLabel/Trainner.yml"):
        messagebox.showerror("Error", "Trained model not found. Please Train Images first.")
        return
    try:
        # --- MAC PATH FIX FOR XML ---
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_cascade = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
        
        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        else:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        # ----------------------------
        
        if os.path.exists("TrainingImageLabel/labels.pkl"):
            with open("TrainingImageLabel/labels.pkl", "rb") as f:
                reverse_label_map = pickle.load(f)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel/Trainner.yml")
        faceCascade = cv2.CascadeClassifier(cascade_path)
        
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            messagebox.showerror("Error", "Cannot access camera")
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        attendance = pd.DataFrame(columns=['Enrollment', 'Name', 'Date', 'Time'])

        marked_students = set()
        face_counter = {}

        # ---- ADD THESE LINES HERE ----
        start_time = time.time()
        last_new_face_time = time.time()
        max_runtime = 7
        # ------------------------------

        while True:
            ret, im = cam.read()
            if not ret: break
            
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80,80))
            
            print("Detected faces:", len(faces))

            for (x, y, w, h) in faces:
                if w < 60 or h < 60:
                    continue
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))

                pred_id, conf = recognizer.predict(face)
                

                # Reject weak matches
                if conf > 65:
                    display_text = "Unknown"
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(im, display_text, (x, y-10), font, 1, (255,255,255), 2)
                    continue

                id_val = reverse_label_map.get(pred_id, "Unknown")

                if id_val != "Unknown": 
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    
                    # Finding the name from the CSV based on the ID
                    name_row = df.loc[df['Enrollment'].astype(str) == str(id_val)]['Name'].values
                    name = name_row[0] if len(name_row) > 0 else "Unknown"
                    
                    accuracy = max(0, min(100, int(100 - conf)))
                    display_text = f"{name} ({accuracy}%)"

                    # Reset all counters except current prediction
                    # Smooth counting (no harsh reset)
                    face_counter[id_val] = face_counter.get(id_val, 0) + 1

                    # Reduce noise gradually
                    

                    # Only mark if SAME face is stable for long
                    if face_counter[id_val] >= 3 and id_val not in marked_students:
                        print(f"Marked: {id_val} - {name}")
                        marked_students.add(id_val)
                        attendance.loc[len(attendance)] = [id_val, name, date, timeStamp]
                        last_new_face_time = time.time()
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    display_text = "Unknown"
                    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                cv2.putText(im, display_text, (x, y-10), font, 1, (255, 255, 255), 2)
                
            cv2.imshow('Attendance System (Press Q to Exit)', im)
            # Stop if no new faces for 3 seconds
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if time.time() - last_new_face_time > 6:
                break

            # Safety stop (maximum runtime)
            if time.time() - start_time > max_runtime:
                break
        
        # Save the attendance record
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        fileName = f"Attendance/Attendance_{date}.csv"
        
        if not attendance.empty:

            print("Saving attendance...")
            print("Final attendance data:")
            print(attendance)

            #  ALWAYS APPEND (no filtering)
            if os.path.exists(fileName):
                attendance.to_csv(fileName, mode='a', header=False, index=False)
                Notification.configure(text="Attendance Updated", bg="SpringGreen3")
            else:
                attendance.to_csv(fileName, index=False)
                Notification.configure(text="Attendance Saved", bg="SpringGreen3")

        else:
            print("No attendance captured")
            Notification.configure(text="No Face Detected — Try Again", bg="red")

        
        
    except Exception as e:

        messagebox.showerror("Error", f"Recognition failed: {e}")

    finally:

        if 'cam' in locals() and cam.isOpened():

            cam.release()

        cv2.destroyAllWindows()
    
def manual_attendance():
    sbw = tk.Toplevel(window)
    sbw.title("Manual Attendance Entry")
    sbw.geometry("460x380")
    sbw.configure(bg="#0f172a")
    sbw.resizable(False, False)

    # Modal Header
    tk.Label(
        sbw, text="Manual Attendance", 
        font=("Helvetica", 18, "bold"), fg="#f8fafc", bg="#0f172a"
    ).pack(pady=(20, 4))
    tk.Label(
        sbw, text="Enter student details to log attendance directly", 
        font=("Helvetica", 11), fg="#94a3b8", bg="#0f172a"
    ).pack(pady=(0, 20))

    form_frame = tk.Frame(sbw, bg="#1e293b", padx=20, pady=20, relief="flat", highlightthickness=1, highlightbackground="#334155")
    form_frame.pack(fill="x", padx=30)

    tk.Label(
        form_frame, text="Enrollment / Roll ID:", 
        font=("Helvetica", 11, "bold"), fg="#cbd5e1", bg="#1e293b", anchor="w"
    ).pack(fill="x", pady=(0, 4))
    en_entry = Entry(
        form_frame, font=("Helvetica", 13), bg="#0f172a", fg="#f8fafc", 
        insertbackground="#38bdf8", relief="flat", highlightthickness=1, highlightbackground="#475569"
    )
    en_entry.pack(fill="x", ipady=6, pady=(0, 14))

    tk.Label(
        form_frame, text="Student Full Name:", 
        font=("Helvetica", 11, "bold"), fg="#cbd5e1", bg="#1e293b", anchor="w"
    ).pack(fill="x", pady=(0, 4))
    nm_entry = Entry(
        form_frame, font=("Helvetica", 13), bg="#0f172a", fg="#f8fafc", 
        insertbackground="#38bdf8", relief="flat", highlightthickness=1, highlightbackground="#475569"
    )
    nm_entry.pack(fill="x", ipady=6, pady=(0, 10))

    def save_manual():
        enrollment = en_entry.get().strip()
        name = nm_entry.get().strip()

        if not enrollment or not name:
            messagebox.showwarning("Warning", "Enrollment ID and Student Name are required!")
            return

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

        fileName = f"Attendance/Attendance_{date}.csv"

        # Create file if not exists
        if not os.path.exists(fileName):
            with open(fileName, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Enrollment', 'Name', 'Date', 'Time'])

        # Duplicate check
        if os.path.getsize(fileName) > 0:
            df = pd.read_csv(fileName)
            if enrollment in df['Enrollment'].astype(str).values:
                messagebox.showwarning("Duplicate", f"Attendance for {enrollment} already marked today!")
                return

        # Save attendance
        with open(fileName, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name, date, timeStamp])

        Notification.configure(text=f"Manual Log: {name} Present", bg="#059669")
        messagebox.showinfo("Success", f"Attendance recorded for {name} ({enrollment})")
        sbw.destroy()

    btn_frame = tk.Frame(sbw, bg="#0f172a")
    btn_frame.pack(fill="x", padx=30, pady=20)

    Button(
        btn_frame, text="Save Attendance", command=save_manual, 
        bg="#10b981", fg="white", font=("Helvetica", 12, "bold"), 
        relief="flat", cursor="hand2", padx=20, pady=8
    ).pack(side="left", expand=True, fill="x", padx=(0, 10))

    Button(
        btn_frame, text="Cancel", command=sbw.destroy, 
        bg="#475569", fg="white", font=("Helvetica", 12), 
        relief="flat", cursor="hand2", padx=20, pady=8
    ).pack(side="right", expand=True, fill="x", padx=(10, 0))


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()

def view_attendance():
    import subprocess
    import platform

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    base_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, f"Attendance/Attendance_{date}.csv")

    if os.path.exists(path):
        time.sleep(0.5)
        df = pd.read_csv(path)

        print("\n===== TODAY ATTENDANCE =====")
        print(df)

        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", path])
            else:  # Linux
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")

        messagebox.showinfo("Success", f"{len(df)} records found & opened")

    else:
        messagebox.showwarning("File Not Found", "No attendance record found for today.")

def sync_to_mysql(fileName):
    # Update these with your actual XAMPP/MAMP credentials
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'db': 'attendance_db' # Make sure this DB exists in phpMyAdmin!
    }
    
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        data = pd.read_csv(fileName)
        
        for i, row in data.iterrows():
            sql = "INSERT INTO attendance (enrollment, name, date, time) VALUES (%s, %s, %s, %s)"
            cursor.execute(sql, tuple(row))
        
        conn.commit()
        conn.close()
        print("Successfully synced to MySQL")
    except Exception as e:
        print(f"Error syncing to MySQL: {e}")
        raise e


# --- MAIN GUI LAYOUT ---
window = tk.Tk()
window.lift()                               # Moves window to the top
window.attributes('-topmost', True)        # Keeps it there
window.after_idle(window.attributes, '-topmost', False) # Allows other windows to move on top again
window.title("VisionAttend AI • Desktop Management Suite")
window.geometry('1240x740')
window.configure(background='#0b0f19')
window.protocol("WM_DELETE_WINDOW", on_closing)

# 1. Header Banner
header_frame = tk.Frame(window, bg="#0f172a", highlightthickness=1, highlightbackground="#1e293b", pady=18, padx=30)
header_frame.pack(fill="x", padx=28, pady=(20, 16))

header_content = tk.Frame(header_frame, bg="#0f172a")
header_content.pack(fill="x")

title_lbl = tk.Label(
    header_content, text="VisionAttend AI 2.0", 
    font=("Helvetica", 24, "bold"), fg="#38bdf8", bg="#0f172a"
)
title_lbl.pack(side="left")

subtitle_lbl = tk.Label(
    header_content, text="  •  Automated Face Recognition & Attendance Suite", 
    font=("Helvetica", 14), fg="#94a3b8", bg="#0f172a"
)
subtitle_lbl.pack(side="left", pady=(5, 0))

engine_badge = tk.Label(
    header_content, text="OpenCV LBPH Active", 
    font=("Helvetica", 11, "bold"), fg="#10b981", bg="#064e3b", padx=12, pady=4
)
engine_badge.pack(side="right")

# 2. Main Content Card (Student Input Form)
main_card = tk.Frame(window, bg="#111827", highlightthickness=1, highlightbackground="#1e293b", padx=30, pady=24)
main_card.pack(fill="x", padx=28, pady=(0, 16))

card_title = tk.Label(
    main_card, text="Student Enrollment & Registration", 
    font=("Helvetica", 16, "bold"), fg="#f8fafc", bg="#111827"
)
card_title.pack(anchor="w", pady=(0, 16))

form_row1 = tk.Frame(main_card, bg="#111827")
form_row1.pack(fill="x", pady=6)

tk.Label(
    form_row1, text="Enrollment ID:", 
    font=("Helvetica", 13, "bold"), fg="#cbd5e1", bg="#111827", width=16, anchor="w"
).pack(side="left")

txt = tk.Entry(
    form_row1, font=("Helvetica", 14), bg="#1e293b", fg="#f8fafc", 
    insertbackground="#38bdf8", relief="flat", highlightthickness=1, highlightbackground="#334155"
)
txt.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 14))

btn_clear1 = Button(
    form_row1, text="Clear", command=clear, 
    bg="#334155", fg="#f8fafc", font=("Helvetica", 11, "bold"), 
    relief="flat", cursor="hand2", padx=18, pady=6
)
btn_clear1.pack(side="left")

form_row2 = tk.Frame(main_card, bg="#111827")
form_row2.pack(fill="x", pady=6)

tk.Label(
    form_row2, text="Student Name:", 
    font=("Helvetica", 13, "bold"), fg="#cbd5e1", bg="#111827", width=16, anchor="w"
).pack(side="left")

txt2 = tk.Entry(
    form_row2, font=("Helvetica", 14), bg="#1e293b", fg="#f8fafc", 
    insertbackground="#38bdf8", relief="flat", highlightthickness=1, highlightbackground="#334155"
)
txt2.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 14))

btn_clear2 = Button(
    form_row2, text="Clear", command=clear1, 
    bg="#334155", fg="#f8fafc", font=("Helvetica", 11, "bold"), 
    relief="flat", cursor="hand2", padx=18, pady=6
)
btn_clear2.pack(side="left")

# 3. Status Notification Bar
status_frame = tk.Frame(window, bg="#0b0f19")
status_frame.pack(fill="x", padx=28, pady=(4, 16))

Notification = Label(
    status_frame, text="System Status: Ready", 
    bg="#1e293b", fg="#38bdf8", font=("Helvetica", 13, "bold"), 
    padx=20, pady=10, relief="flat", highlightthickness=1, highlightbackground="#334155"
)
Notification.pack(fill="x")

# 4. Action Command Buttons (Two-Row Structured Grid)
actions_frame = tk.Frame(window, bg="#0b0f19")
actions_frame.pack(fill="x", padx=28, pady=4)

# Row 1: Dataset & AI Training
row1_frame = tk.Frame(actions_frame, bg="#0b0f19")
row1_frame.pack(fill="x", pady=6)

btn_take = Button(
    row1_frame, text="📷  Take Face Samples", command=take_img, 
    bg="#2563eb", fg="white", font=("Helvetica", 13, "bold"), 
    relief="flat", cursor="hand2", pady=14
)
btn_take.pack(side="left", expand=True, fill="x", padx=(0, 10))

btn_train = Button(
    row1_frame, text="🧠  Train AI Model", command=trainimg, 
    bg="#7c3aed", fg="white", font=("Helvetica", 13, "bold"), 
    relief="flat", cursor="hand2", pady=14
)
btn_train.pack(side="left", expand=True, fill="x", padx=10)

btn_auto = Button(
    row1_frame, text="⚡  Automatic Attendance", command=automatic_attendance, 
    bg="#059669", fg="white", font=("Helvetica", 13, "bold"), 
    relief="flat", cursor="hand2", pady=14
)
btn_auto.pack(side="left", expand=True, fill="x", padx=(10, 0))

# Row 2: Attendance Records & Controls
row2_frame = tk.Frame(actions_frame, bg="#0b0f19")
row2_frame.pack(fill="x", pady=6)

btn_manual = Button(
    row2_frame, text="✍️  Manual Attendance", command=manual_attendance, 
    bg="#334155", fg="white", font=("Helvetica", 13, "bold"), 
    relief="flat", cursor="hand2", pady=14
)
btn_manual.pack(side="left", expand=True, fill="x", padx=(0, 10))

btn_view = Button(
    row2_frame, text="📋  View Today's Attendance", command=view_attendance, 
    bg="#0284c7", fg="white", font=("Helvetica", 13, "bold"), 
    relief="flat", cursor="hand2", pady=14
)
btn_view.pack(side="left", expand=True, fill="x", padx=10)

btn_quit = Button(
    row2_frame, text="✕  Quit System", command=on_closing, 
    bg="#dc2626", fg="white", font=("Helvetica", 13, "bold"), 
    relief="flat", cursor="hand2", pady=14
)
btn_quit.pack(side="left", expand=True, fill="x", padx=(10, 0))

window.mainloop()