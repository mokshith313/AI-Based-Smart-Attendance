document.addEventListener('DOMContentLoaded', () => {
    // --- APP STATE ---
    let activeWebcamStream = null;
    let regWebcamStream = null;
    let isRecognizing = false;
    let recognitionInterval = null;
    let capturedFrames = [];

    // --- DOM ELEMENTS ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    const notificationBanner = document.getElementById('notification-banner');
    const notificationText = document.getElementById('notification-text');
    const btnCloseNotif = document.getElementById('btn-close-notif');

    const statRegistered = document.getElementById('stat-registered-students');
    const statPresent = document.getElementById('stat-present-today');
    const statTrained = document.getElementById('stat-trained-samples');
    const statModelStatus = document.getElementById('stat-model-status');
    const btnRefreshStats = document.getElementById('btn-refresh-stats');

    // Tab 1 Elements (Live Recognition)
    const btnToggleCam = document.getElementById('btn-toggle-cam');
    const webcamVideo = document.getElementById('webcam-video');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const camPlaceholder = document.getElementById('cam-placeholder');
    const recognitionFeed = document.getElementById('recognition-feed');
    const miniAttendanceList = document.getElementById('mini-attendance-list');
    const btnTrainModel = document.getElementById('btn-train-model');
    const btnTrainModelSec = document.getElementById('btn-train-model-sec');

    // Tab 2 Elements (Registration)
    const formRegister = document.getElementById('form-register-student');
    const inputRegEnrollment = document.getElementById('reg-enrollment');
    const inputRegName = document.getElementById('reg-name');
    const btnStartRegCam = document.getElementById('btn-start-reg-cam');
    const btnCaptureFaces = document.getElementById('btn-capture-faces');
    const regWebcamVideo = document.getElementById('reg-webcam-video');
    const regCamPlaceholder = document.getElementById('reg-cam-placeholder');
    const captureProgressWrap = document.getElementById('capture-progress-wrap');
    const captureCountText = document.getElementById('capture-count-text');
    const captureProgressBar = document.getElementById('capture-progress-bar');
    const capturedThumbnails = document.getElementById('captured-thumbnails');

    // Tab 3 Elements (Manual Entry)
    const formManual = document.getElementById('form-manual-attendance');
    const inputManEnrollment = document.getElementById('man-enrollment');
    const inputManName = document.getElementById('man-name');

    // Tab 4 Elements (Logs)
    const attendanceDatePicker = document.getElementById('attendance-date-picker');
    const btnFetchLogs = document.getElementById('btn-fetch-logs');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const tbodyAttendance = document.getElementById('tbody-attendance');

    // Tab 5 Elements (Students)
    const tbodyStudents = document.getElementById('tbody-students');

    // Set today's date in datepicker
    const todayISO = new Date().toISOString().split('T')[0];
    if (attendanceDatePicker) attendanceDatePicker.value = todayISO;

    // --- INITIALIZATION ---
    fetchStats();
    fetchTodayLogs();
    fetchStudentDirectory();

    // --- TAB SWITCHING ---
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'tab-logs') fetchAttendanceLogs(attendanceDatePicker.value);
            if (targetTab === 'tab-students') fetchStudentDirectory();
        });
    });

    // --- NOTIFICATION HELPER ---
    function showNotification(message, type = 'info') {
        notificationText.textContent = message;
        notificationBanner.className = `notification ${type}`;
        notificationBanner.classList.remove('hidden');

        setTimeout(() => {
            notificationBanner.classList.add('hidden');
        }, 6000);
    }

    if (btnCloseNotif) {
        btnCloseNotif.addEventListener('click', () => {
            notificationBanner.classList.add('hidden');
        });
    }

    // --- STATS FETCHING ---
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.success) {
                statRegistered.textContent = data.total_students;
                statPresent.textContent = data.today_present;
                statTrained.textContent = data.trained_images_count;
                if (data.model_ready) {
                    statModelStatus.textContent = "Ready";
                    statModelStatus.className = "stat-value model-badge";
                } else {
                    statModelStatus.textContent = "Not Trained";
                    statModelStatus.className = "stat-value text-muted";
                }
            }
        } catch (err) {
            console.error('Error fetching stats:', err);
        }
    }

    if (btnRefreshStats) {
        btnRefreshStats.addEventListener('click', () => {
            fetchStats();
            fetchTodayLogs();
            fetchStudentDirectory();
            showNotification('Dashboard refreshed', 'success');
        });
    }

    // --- TAB 1: LIVE RECOGNITION CAMERA ---
    if (btnToggleCam) {
        btnToggleCam.addEventListener('click', async () => {
            if (isRecognizing) {
                stopRecognitionCamera();
            } else {
                await startRecognitionCamera();
            }
        });
    }

    async function startRecognitionCamera() {
        try {
            activeWebcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
            });
            webcamVideo.srcObject = activeWebcamStream;
            camPlaceholder.classList.add('hidden');

            isRecognizing = true;
            btnToggleCam.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Camera';
            btnToggleCam.className = 'btn btn-secondary';

            // Start sending frames to backend recognizer
            recognitionInterval = setInterval(processRecognitionFrame, 600);
            showNotification('Live camera scanning activated', 'success');
        } catch (err) {
            console.error('Camera access error:', err);
            showNotification('Cannot access webcam: ' + err.message, 'error');
        }
    }

    function stopRecognitionCamera() {
        if (recognitionInterval) clearInterval(recognitionInterval);
        if (activeWebcamStream) {
            activeWebcamStream.getTracks().forEach(track => track.stop());
            activeWebcamStream = null;
        }

        webcamVideo.srcObject = null;
        camPlaceholder.classList.remove('hidden');
        clearCanvasOverlay();

        isRecognizing = false;
        btnToggleCam.innerHTML = '<i class="fa-solid fa-play"></i> Start Camera';
        btnToggleCam.className = 'btn btn-primary';
    }

    async function processRecognitionFrame() {
        if (!isRecognizing || !webcamVideo.videoWidth) return;

        // Capture current video frame to base64
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        const frameBase64 = canvas.toDataURL('image/jpeg', 0.85);

        try {
            const res = await fetch('/api/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frameBase64 })
            });

            const data = await res.json();
            if (data.success) {
                drawBoundingBoxes(data.faces, canvas.width, canvas.height);
                updateRecognitionFeed(data.faces);
            } else if (data.error && data.error.includes('Trained model not found')) {
                stopRecognitionCamera();
                showNotification(data.error, 'error');
            }
        } catch (err) {
            console.error('Frame recognition error:', err);
        }
    }

    function clearCanvasOverlay() {
        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }

    function drawBoundingBoxes(faces, videoW, videoH) {
        overlayCanvas.width = webcamVideo.clientWidth;
        overlayCanvas.height = webcamVideo.clientHeight;

        const ctx = overlayCanvas.getContext('2d');
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        const scaleX = overlayCanvas.width / videoW;
        const scaleY = overlayCanvas.height / videoH;

        faces.forEach(face => {
            const [x, y, w, h] = face.bbox;
            const rx = x * scaleX;
            const ry = y * scaleY;
            const rw = w * scaleX;
            const rh = h * scaleY;

            ctx.lineWidth = 3;
            if (face.is_known) {
                ctx.strokeStyle = face.is_marked ? '#10b981' : '#3b82f6';
                ctx.fillStyle = face.is_marked ? 'rgba(16, 185, 129, 0.2)' : 'rgba(59, 130, 246, 0.2)';
            } else {
                ctx.strokeStyle = '#ef4444';
                ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
            }

            ctx.fillRect(rx, ry, rw, rh);
            ctx.strokeRect(rx, ry, rw, rh);

            // Text Label Box
            const label = face.is_known ? `${face.name} (${face.accuracy}%)` : 'Unknown Face';
            ctx.font = 'bold 14px Outfit, sans-serif';
            const textWidth = ctx.measureText(label).width;

            ctx.fillStyle = face.is_known ? (face.is_marked ? '#10b981' : '#3b82f6') : '#ef4444';
            ctx.fillRect(rx, ry - 28 > 0 ? ry - 28 : ry, textWidth + 16, 26);

            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, rx + 8, ry - 28 > 0 ? ry - 10 : ry + 18);

            if (face.just_marked) {
                showNotification(`Attendance Recorded for ${face.name}!`, 'success');
                fetchStats();
                fetchTodayLogs();
            }
        });
    }

    function updateRecognitionFeed(faces) {
        if (!faces || faces.length === 0) return;

        let feedHtml = '';
        faces.forEach(face => {
            if (face.is_known) {
                feedHtml += `
                    <div class="feed-item ${face.is_marked ? 'marked' : ''}">
                        <div>
                            <div class="feed-name">${face.name}</div>
                            <div class="feed-meta">ID: ${face.enrollment} &bull; Accuracy: ${face.accuracy}%</div>
                        </div>
                        <span class="feed-badge ${face.is_marked ? 'success' : ''}">
                            ${face.is_marked ? '<i class="fa-solid fa-check"></i> Present' : 'Detected'}
                        </span>
                    </div>
                `;
            } else {
                feedHtml += `
                    <div class="feed-item">
                        <div>
                            <div class="feed-name">Unknown Face</div>
                            <div class="feed-meta">Unrecognized sample</div>
                        </div>
                        <span class="feed-badge unknown">Unknown</span>
                    </div>
                `;
            }
        });

        recognitionFeed.innerHTML = feedHtml;
    }

    async function fetchTodayLogs() {
        try {
            const res = await fetch('/api/attendance');
            const data = await res.json();
            if (data.success && data.records) {
                if (data.records.length === 0) {
                    miniAttendanceList.innerHTML = '<p class="text-muted">No attendance marked yet today</p>';
                    return;
                }

                let html = '';
                // Render last 5 entries
                const recent = data.records.slice(-5).reverse();
                recent.forEach(item => {
                    html += `
                        <div class="mini-log-item">
                            <span><strong>${item.Name}</strong> (${item.Enrollment})</span>
                            <span class="text-muted">${item.Time}</span>
                        </div>
                    `;
                });
                miniAttendanceList.innerHTML = html;
            }
        } catch (err) {
            console.error('Error fetching today logs:', err);
        }
    }

    // --- MODEL TRAINING ---
    async function triggerModelTraining() {
        showNotification('Training face recognition model... Please wait...', 'info');
        try {
            const res = await fetch('/api/train', { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                showNotification(data.message, 'success');
                fetchStats();
            } else {
                showNotification('Training failed: ' + data.error, 'error');
            }
        } catch (err) {
            showNotification('Server error during model training', 'error');
        }
    }

    if (btnTrainModel) btnTrainModel.addEventListener('click', triggerModelTraining);
    if (btnTrainModelSec) btnTrainModelSec.addEventListener('click', triggerModelTraining);

    // --- TAB 2: STUDENT REGISTRATION ---
    if (btnStartRegCam) {
        btnStartRegCam.addEventListener('click', async () => {
            if (regWebcamStream) {
                stopRegCamera();
            } else {
                await startRegCamera();
            }
        });
    }

    async function startRegCamera() {
        try {
            regWebcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
            regWebcamVideo.srcObject = regWebcamStream;
            regCamPlaceholder.classList.add('hidden');

            btnStartRegCam.innerHTML = '<i class="fa-solid fa-stop"></i> Close Camera';
            btnCaptureFaces.disabled = false;
        } catch (err) {
            showNotification('Cannot access camera: ' + err.message, 'error');
        }
    }

    function stopRegCamera() {
        if (regWebcamStream) {
            regWebcamStream.getTracks().forEach(track => track.stop());
            regWebcamStream = null;
        }
        regWebcamVideo.srcObject = null;
        regCamPlaceholder.classList.remove('hidden');

        btnStartRegCam.innerHTML = '<i class="fa-solid fa-camera"></i> Open Camera';
        btnCaptureFaces.disabled = true;
    }

    if (btnCaptureFaces) {
        btnCaptureFaces.addEventListener('click', async () => {
            const enrollment = inputRegEnrollment.value.trim();
            const name = inputRegName.value.trim();

            if (!enrollment || !name) {
                showNotification('Please enter student Enrollment ID and Name first', 'error');
                return;
            }

            btnCaptureFaces.disabled = true;
            captureProgressWrap.classList.remove('hidden');
            capturedThumbnails.innerHTML = '';
            capturedFrames = [];

            const totalSamples = 15;
            for (let i = 1; i <= totalSamples; i++) {
                captureCountText.textContent = `${i} / ${totalSamples}`;
                captureProgressBar.style.width = `${(i / totalSamples) * 100}%`;

                // Grab frame from video
                const canvas = document.createElement('canvas');
                canvas.width = regWebcamVideo.videoWidth || 640;
                canvas.height = regWebcamVideo.videoHeight || 480;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(regWebcamVideo, 0, 0, canvas.width, canvas.height);

                const frameData = canvas.toDataURL('image/jpeg', 0.85);
                capturedFrames.push(frameData);

                // Add thumbnail preview
                if (i <= 8) {
                    const imgThumb = document.createElement('img');
                    imgThumb.src = frameData;
                    imgThumb.className = 'captured-thumb';
                    capturedThumbnails.appendChild(imgThumb);
                }

                await new Promise(r => setTimeout(r, 120)); // delay between captures
            }

            // Submit registration payload
            showNotification('Saving student profile and face dataset...', 'info');
            try {
                const res = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        enrollment: enrollment,
                        name: name,
                        images: capturedFrames
                    })
                });

                const data = await res.json();
                if (data.success) {
                    showNotification(data.message, 'success');
                    inputRegEnrollment.value = '';
                    inputRegName.value = '';
                    stopRegCamera();
                    fetchStats();
                    fetchStudentDirectory();

                    // Prompt training
                    setTimeout(() => {
                        if (confirm('Student registered! Would you like to retrain the AI model now?')) {
                            triggerModelTraining();
                        }
                    }, 500);
                } else {
                    showNotification('Registration error: ' + data.error, 'error');
                }
            } catch (err) {
                showNotification('Failed to register student', 'error');
            } finally {
                captureProgressWrap.classList.add('hidden');
                btnCaptureFaces.disabled = false;
            }
        });
    }

    // --- TAB 3: MANUAL ATTENDANCE FORM ---
    if (formManual) {
        formManual.addEventListener('submit', async (e) => {
            e.preventDefault();
            const enrollment = inputManEnrollment.value.trim();
            const name = inputManName.value.trim();

            if (!enrollment || !name) return;

            try {
                const res = await fetch('/api/manual-attendance', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enrollment, name })
                });

                const data = await res.json();
                if (data.success) {
                    showNotification(data.message, 'success');
                    inputManEnrollment.value = '';
                    inputManName.value = '';
                    fetchStats();
                    fetchTodayLogs();
                } else {
                    showNotification('Manual entry failed: ' + data.error, 'error');
                }
            } catch (err) {
                showNotification('Failed to submit manual attendance', 'error');
            }
        });
    }

    // --- TAB 4: ATTENDANCE LOGS TABLE & CSV ---
    async function fetchAttendanceLogs(dateStr) {
        try {
            tbodyAttendance.innerHTML = '<tr><td colspan="6" class="text-center py-4">Fetching records...</td></tr>';
            const res = await fetch(`/api/attendance?date=${dateStr}`);
            const data = await res.json();

            if (data.success && data.records) {
                if (data.records.length === 0) {
                    tbodyAttendance.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-muted">No attendance entries found for ${dateStr}</td></tr>`;
                    return;
                }

                let html = '';
                data.records.forEach((row, idx) => {
                    html += `
                        <tr>
                            <td>${idx + 1}</td>
                            <td><strong>${row.Enrollment}</strong></td>
                            <td>${row.Name}</td>
                            <td>${row.Date}</td>
                            <td>${row.Time}</td>
                            <td><span class="status-badge present"><i class="fa-solid fa-check"></i> Present</span></td>
                        </tr>
                    `;
                });
                tbodyAttendance.innerHTML = html;
            }
        } catch (err) {
            tbodyAttendance.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">Failed to load attendance logs</td></tr>';
        }
    }

    if (btnFetchLogs) {
        btnFetchLogs.addEventListener('click', () => {
            fetchAttendanceLogs(attendanceDatePicker.value);
        });
    }

    if (btnExportCsv) {
        btnExportCsv.addEventListener('click', () => {
            const dateVal = attendanceDatePicker.value || todayISO;
            window.location.href = `/api/export-attendance?date=${dateVal}`;
        });
    }

    // --- TAB 5: STUDENT DIRECTORY TABLE ---
    async function fetchStudentDirectory() {
        try {
            tbodyStudents.innerHTML = '<tr><td colspan="5" class="text-center py-4">Fetching registered students...</td></tr>';
            const res = await fetch('/api/students');
            const data = await res.json();

            if (data.success && data.students) {
                if (data.students.length === 0) {
                    tbodyStudents.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">No registered students yet</td></tr>';
                    return;
                }

                let html = '';
                data.students.forEach((s, idx) => {
                    html += `
                        <tr>
                            <td>${idx + 1}</td>
                            <td><strong>${s.Enrollment}</strong></td>
                            <td>${s.Name}</td>
                            <td>${s.Date || 'N/A'}</td>
                            <td>${s.Time || 'N/A'}</td>
                        </tr>
                    `;
                });
                tbodyStudents.innerHTML = html;
            }
        } catch (err) {
            tbodyStudents.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">Failed to load student directory</td></tr>';
        }
    }
});
