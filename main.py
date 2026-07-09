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
    sbw.title("Manual Entry")
    sbw.geometry("400x300")

    Label(sbw, text="Enrollment:").pack(pady=5)
    en_entry = Entry(sbw)
    en_entry.pack()

    Label(sbw, text="Name:").pack(pady=5)
    nm_entry = Entry(sbw)
    nm_entry.pack()

    def save_manual():
        enrollment = en_entry.get()
        name = nm_entry.get()

        if not enrollment or not name:
            messagebox.showwarning("Warning", "All fields required!")
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
                messagebox.showwarning("Duplicate", "Already marked today!")
                return

        # Save attendance
        with open(fileName, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([enrollment, name, date, timeStamp])

        messagebox.showinfo("Success", "Manual attendance saved")
        sbw.destroy()

    #  IMPORTANT BUTTON (you missed earlier)
    Button(sbw, text="Save", command=save_manual).pack(pady=20)


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
window.title("FAMS-Face Recognition System")
window.geometry('1280x720')
window.configure(background='grey80')
window.protocol("WM_DELETE_WINDOW", on_closing)

Label(window, text="Attendance Management System", bg="black", fg="white", width=50, height=3, font=('times', 30, 'bold')).place(x=80, y=20)
Notification = Label(window, text="Status: Ready", bg="Green", fg="white", width=30, height=2, font=('times', 17))
Notification.place(x=450, y=400)

# --- Replace your current Entry boxes with these ---

Label(window, text="Enter Enrollment:", width=20, bg="grey", font=('times', 15, 'bold')).place(x=200, y=200)

# Updated txt Entry
txt = tk.Entry(window, width=20, bg="white", fg="black", 
               insertbackground='black', 
               font=('times', 25), highlightthickness=2)
txt.place(x=400, y=210)

Label(window, text="Enter Name:", width=20, bg="grey", font=('times', 15, 'bold')).place(x=200, y=300)

# Updated txt2 Entry
txt2 = tk.Entry(window, width=20, bg="white", fg="black", 
                insertbackground='black', 
                font=('times', 25), highlightthickness=2)
txt2.place(x=400, y=310)

Button(window, text="Clear", command=clear, fg="white", bg="black", width=10).place(x=950, y=210)
Button(window, text="Clear", command=clear1, fg="white", bg="black", width=10).place(x=950, y=310)

Button(window, text="Take Images", command=take_img, bg="SkyBlue1", width=20, height=3, font=('times', 15, 'bold')).place(x=90, y=500)
Button(window, text="Train Images", command=trainimg, bg="SkyBlue1", width=20, height=3, font=('times', 15, 'bold')).place(x=390, y=500)
Button(window, text="Automatic Attendance", command=automatic_attendance, bg="SkyBlue1", width=20, height=3, font=('times', 15, 'bold')).place(x=690, y=500)
Button(window, text="Manual Attendance", command=manual_attendance, bg="SkyBlue1", width=20, height=3, font=('times', 15, 'bold')).place(x=90, y=620)
Button(window, text="Quit System", command=on_closing, bg="red3", fg="white", width=20, height=3, font=('times', 15, 'bold')).place(x=690, y=620)
tk.Button(window, text="View Attendance", command=view_attendance, 
          bg="SkyBlue1", width=20, height=3, font=('times', 15, 'bold')).place(x=390, y=620)
window.mainloop()