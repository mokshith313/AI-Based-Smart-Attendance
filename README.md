# AI-Based Smart Attendance System

An intelligent, real-time face recognition attendance management system equipped with a modern web interface ready for one-click deployment on **Render** (as well as desktop support via Tkinter).

---

## 🚀 Deployment on Render (Step-by-Step Guide)

You can deploy this application directly to [Render](https://render.com) using either **Native Web Service** or **Docker Container**.

### Option 1: Web Service (Render Blueprint - Recommended)

1. **Push your code to GitHub / GitLab**:
   Ensure your repository includes all generated project files: `requirements.txt`, `app.py`, `render.yaml`, `Procfile`, `apt.txt`, `haarcascade_frontalface_default.xml`, `templates/`, and `static/`.

2. **Connect to Render**:
   - Go to your [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** -> Select **Blueprint**.
   - Connect your GitHub repository containing this project.
   - Render will automatically pick up `render.yaml` and configure:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
     - **Environment Variables**: `PYTHON_VERSION=3.10.12`

3. **Deploy**:
   - Click **Apply**. Render will install dependencies and launch your application!

---

### Option 2: Docker Deployment on Render

If you prefer containerized deployment (guaranteeing exact OpenCV C-libraries):

1. Go to [Render Dashboard](https://dashboard.render.com/) -> **New +** -> **Web Service**.
2. Connect your repository.
3. Select **Docker** as the Runtime environment.
4. Render will automatically detect the included `Dockerfile`, build the container, and launch `gunicorn app:app`.

---

## 💻 Running Locally

### Option A: Web Application (`app.py`)
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the local Flask web server:
   ```bash
   python app.py
   ```
3. Open your browser at `http://localhost:5000`.

### Option B: Desktop Application (`main.py`)
If running on a desktop with a connected USB camera and GUI display:
```bash
pip install opencv-contrib-python pandas pillow pymysql
python main.py
```

---

## 📁 Repository Structure

```
├── app.py                            # Flask Web application (Render production server)
├── main.py                           # Desktop Tkinter GUI application
├── requirements.txt                  # Python dependencies for Render
├── render.yaml                       # Render Blueprint manifest
├── Procfile                          # Gunicorn start process for cloud host
├── Dockerfile                        # Docker configuration for Render Docker environment
├── apt.txt                           # Linux system dependencies for Render buildpack
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade face detector model
├── StudentDetails/                   # CSV storage for registered student details
├── TrainingImage/                    # Saved face images dataset for training
├── TrainingImageLabel/               # Trained LBPH model weights (.yml & .pkl)
├── Attendance/                       # Daily attendance CSV log files
├── static/                           # Web CSS styles & JS logic
│   ├── css/style.css
│   └── js/main.js
└── templates/                        # Web HTML dashboard layout
    └── index.html
```

---

## 🌟 Key Features

- 📹 **Live Web Camera Face Scanner**: Real-time multi-face detection, identification, and bounding box visualization in browser canvas.
- 👤 **Student Registration**: Capture 15–50 face images directly via browser webcam.
- 🧠 **On-Demand AI Training**: Trains OpenCV LBPH (Local Binary Patterns Histograms) Face Recognizer model on the cloud server.
- 📊 **Automatic Attendance Logging**: Anti-duplicate logic automatically logs present students to date-stamped CSV files.
- ✍️ **Manual Attendance**: Option to record manual attendance entries for edge cases.
- 📥 **CSV Export**: One-click download of attendance reports per date.
- 🎨 **Glassmorphism UI**: High-performance dark-mode web dashboard.
