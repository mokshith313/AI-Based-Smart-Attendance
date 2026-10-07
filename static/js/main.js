/**
 * VisionAttend AI - Frontend Application Controller
 * Real-Time Face Recognition, Attendance Logging, and Dashboard UI
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- APP STATE ---
    let activeWebcamStream = null;
    let regWebcamStream = null;
    let isRecognizing = false;
    let recognitionInterval = null;
    let capturedFrames = [];

    // Cache of fetched records for instant client-side searching
    let cachedAttendanceRecords = [];
    let cachedStudentDirectory = [];

    // Audio Chime Synthesizer Context (Web Audio API)
    let audioCtx = null;
    function playVerificationChime() {
        try {
            if (!audioCtx) {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }
            // Double-tone pleasant success chord (E5 -> G#5)
            const now = audioCtx.currentTime;
            
            const osc1 = audioCtx.createOscillator();
            const gain1 = audioCtx.createGain();
            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(659.25, now); // E5
            gain1.gain.setValueAtTime(0.12, now);
            gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.25);
            osc1.connect(gain1);
            gain1.connect(audioCtx.destination);
            osc1.start(now);
            osc1.stop(now + 0.25);

            const osc2 = audioCtx.createOscillator();
            const gain2 = audioCtx.createGain();
            osc2.type = 'sine';
            osc2.frequency.setValueAtTime(830.61, now + 0.1); // G#5
            gain2.gain.setValueAtTime(0.15, now + 0.1);
            gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
            osc2.connect(gain2);
            gain2.connect(audioCtx.destination);
            osc2.start(now + 0.1);
            osc2.stop(now + 0.4);
        } catch (e) {
            // Audio not supported or blocked by browser policy
        }
    }

    // --- DOM ELEMENTS ---
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    const toastContainer = document.getElementById('toast-container');
    const notificationBanner = document.getElementById('notification-banner');
    const notificationText = document.getElementById('notification-text');
    const btnCloseNotif = document.getElementById('btn-close-notif');

    // Header Clock Elements
    const clockTimeElem = document.getElementById('clock-time');
    const clockDateElem = document.getElementById('clock-date');

    // Stats Elements
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
    const camViewportContainer = document.getElementById('cam-viewport-container');
    const hudCamStatus = document.getElementById('hud-cam-status');
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

    // Tab 4 Elements (Attendance Logs)
    const attendanceDatePicker = document.getElementById('attendance-date-picker');
    const btnFetchLogs = document.getElementById('btn-fetch-logs');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const tbodyAttendance = document.getElementById('tbody-attendance');
    const inputSearchLogs = document.getElementById('input-search-logs');
    const logsCountChip = document.getElementById('logs-count-chip');

    // Tab 5 Elements (Student Directory)
    const tbodyStudents = document.getElementById('tbody-students');
    const inputSearchStudents = document.getElementById('input-search-students');
    const studentsCountChip = document.getElementById('students-count-chip');

    // Date Picker Default Today
    const todayISO = new Date().toISOString().split('T')[0];
    if (attendanceDatePicker) attendanceDatePicker.value = todayISO;

    // --- DIGITAL CLOCK SYSTEM ---
    function updateClock() {
        const now = new Date();
        if (clockTimeElem) {
            clockTimeElem.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
        if (clockDateElem) {
            clockDateElem.textContent = now.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
        }
    }
    updateClock();
    setInterval(updateClock, 1000);

    // --- AVATAR INITIALS HELPER ---
    function getInitials(name) {
        if (!name || name === 'Unknown' || name === 'Unknown Face') return '?';
        const parts = name.trim().split(/\s+/);
        if (parts.length === 1) return parts[0].substring(0, 2).toUpperCase();
        return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }

    // --- TOAST NOTIFICATION ENGINE ---
    function showNotification(message, type = 'info') {
        // Fallback banner element compatibility
        if (notificationText && notificationBanner) {
            notificationText.textContent = message;
            notificationBanner.className = `notification ${type}`;
        }

        // Modern floating toast stack
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        let iconClass = 'fa-info-circle';
        let title = 'Information';
        if (type === 'success') {
            iconClass = 'fa-circle-check';
            title = 'Success';
        } else if (type === 'error') {
            iconClass = 'fa-circle-exclamation';
            title = 'Attention Required';
        } else if (type === 'warning') {
            iconClass = 'fa-triangle-exclamation';
            title = 'Notice';
        }

        toast.innerHTML = `
            <i class="fa-solid ${iconClass} toast-icon"></i>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" aria-label="Close Notification">
                <i class="fa-solid fa-xmark"></i>
            </button>
        `;

        const closeBtn = toast.querySelector('.toast-close');
        closeBtn.addEventListener('click', () => removeToast(toast));

        toastContainer.appendChild(toast);

        // Auto remove after 4.5 seconds
        const timeout = setTimeout(() => {
            removeToast(toast);
        }, 4500);

        function removeToast(el) {
            clearTimeout(timeout);
            el.classList.add('removing');
            setTimeout(() => {
                if (el.parentNode) el.parentNode.removeChild(el);
            }, 300);
        }
    }

    if (btnCloseNotif) {
        btnCloseNotif.addEventListener('click', () => {
            notificationBanner.classList.add('hidden');
        });
    }

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
            const targetPanel = document.getElementById(targetTab);
            if (targetPanel) targetPanel.classList.add('active');

            if (targetTab === 'tab-logs') fetchAttendanceLogs(attendanceDatePicker.value);
            if (targetTab === 'tab-students') fetchStudentDirectory();
        });
    });

    // --- STATS FETCHING ---
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            if (data.success) {
                if (statRegistered) statRegistered.textContent = data.total_students;
                if (statPresent) statPresent.textContent = data.today_present;
                if (statTrained) statTrained.textContent = data.trained_images_count;
                if (statModelStatus) {
                    if (data.model_ready) {
                        statModelStatus.innerHTML = '<i class="fa-solid fa-circle-check"></i> Trained & Ready';
                        statModelStatus.className = "stat-value model-badge";
                    } else {
                        statModelStatus.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Untrained';
                        statModelStatus.className = "stat-value model-badge not-trained";
                    }
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
            showNotification('Dashboard statistics updated', 'info');
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
            if (camViewportContainer) camViewportContainer.classList.add('scanning');

            if (hudCamStatus) {
                hudCamStatus.className = 'hud-live-badge active';
                hudCamStatus.innerHTML = '<span class="pulse-dot green"></span><span>CAM LIVE</span>';
            }

            isRecognizing = true;
            btnToggleCam.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Scanner';
            btnToggleCam.className = 'btn btn-danger';

            // Start sending frames to backend recognizer every 600ms
            recognitionInterval = setInterval(processRecognitionFrame, 600);
            showNotification('Live face scanner activated', 'success');
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
        if (camViewportContainer) camViewportContainer.classList.remove('scanning');

        if (hudCamStatus) {
            hudCamStatus.className = 'hud-live-badge inactive';
            hudCamStatus.innerHTML = '<span class="pulse-dot green"></span><span>CAM STANDBY</span>';
        }

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
            // Video has scaleX(-1) for mirror view, so mirror X coordinates for overlay canvas
            const mirroredX = videoW - (x + w);
            const rx = mirroredX * scaleX;
            const ry = y * scaleY;
            const rw = w * scaleX;
            const rh = h * scaleY;

            let strokeColor = '#f43f5e'; // red
            let fillColor = 'rgba(244, 63, 94, 0.12)';
            let tagColor = '#f43f5e';

            if (face.is_known) {
                if (face.is_marked) {
                    strokeColor = '#10b981'; // emerald
                    fillColor = 'rgba(16, 185, 129, 0.15)';
                    tagColor = '#10b981';
                } else {
                    strokeColor = '#6366f1'; // indigo
                    fillColor = 'rgba(99, 102, 241, 0.15)';
                    tagColor = '#6366f1';
                }
            }

            // Draw bounding box with rounded corners
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = strokeColor;
            ctx.fillStyle = fillColor;

            const radius = 8;
            ctx.beginPath();
            ctx.moveTo(rx + radius, ry);
            ctx.lineTo(rx + rw - radius, ry);
            ctx.quadraticCurveTo(rx + rw, ry, rx + rw, ry + radius);
            ctx.lineTo(rx + rw, ry + rh - radius);
            ctx.quadraticCurveTo(rx + rw, ry + rh, rx + rw - radius, ry + rh);
            ctx.lineTo(rx + radius, ry + rh);
            ctx.quadraticCurveTo(rx, ry + rh, rx, ry + rh - radius);
            ctx.lineTo(rx, ry + radius);
            ctx.quadraticCurveTo(rx, ry, rx + radius, ry);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // Tactical Corner Accents
            const cornerLen = 14;
            ctx.lineWidth = 3.5;
            ctx.strokeStyle = strokeColor;
            
            // Top Left Corner
            ctx.beginPath();
            ctx.moveTo(rx, ry + cornerLen);
            ctx.lineTo(rx, ry);
            ctx.lineTo(rx + cornerLen, ry);
            ctx.stroke();

            // Top Right Corner
            ctx.beginPath();
            ctx.moveTo(rx + rw - cornerLen, ry);
            ctx.lineTo(rx + rw, ry);
            ctx.lineTo(rx + rw, ry + cornerLen);
            ctx.stroke();

            // Bottom Left Corner
            ctx.beginPath();
            ctx.moveTo(rx, ry + rh - cornerLen);
            ctx.lineTo(rx, ry + rh);
            ctx.lineTo(rx + cornerLen, ry + rh);
            ctx.stroke();

            // Bottom Right Corner
            ctx.beginPath();
            ctx.moveTo(rx + rw - cornerLen, ry + rh);
            ctx.lineTo(rx + rw, ry + rh);
            ctx.lineTo(rx + rw, ry + rh - cornerLen);
            ctx.stroke();

            // Label Tag Pill
            const label = face.is_known 
                ? `${face.name} • ${face.accuracy}%` 
                : 'Unknown Face';
            
            ctx.font = '600 13px Outfit, sans-serif';
            const textMetrics = ctx.measureText(label);
            const tagW = textMetrics.width + 20;
            const tagH = 26;
            const tagY = ry - tagH - 4 > 4 ? ry - tagH - 4 : ry + rh + 4;

            // Draw Tag Background
            ctx.fillStyle = tagColor;
            ctx.beginPath();
            ctx.roundRect(rx, tagY, tagW, tagH, 6);
            ctx.fill();

            // Draw Tag Text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, rx + 10, tagY + 18);

            // Trigger attendance celebration if just marked
            if (face.just_marked) {
                playVerificationChime();
                showNotification(`Verified & Recorded: ${face.name}!`, 'success');
                fetchStats();
                fetchTodayLogs();
            }
        });
    }

    function updateRecognitionFeed(faces) {
        if (!faces || faces.length === 0) return;

        let feedHtml = '';
        faces.forEach(face => {
            const avatarInitials = getInitials(face.name);
            if (face.is_known) {
                feedHtml += `
                    <div class="feed-item ${face.is_marked ? 'marked' : ''}">
                        <div class="feed-info">
                            <div class="feed-avatar">${avatarInitials}</div>
                            <div>
                                <div class="feed-name">${face.name}</div>
                                <div class="feed-meta">ID: <strong>${face.enrollment}</strong> &bull; Match: ${face.accuracy}%</div>
                            </div>
                        </div>
                        <span class="feed-badge ${face.is_marked ? 'success' : 'detecting'}">
                            ${face.is_marked ? '<i class="fa-solid fa-check"></i> Present' : '<i class="fa-solid fa-crosshairs"></i> Tracking'}
                        </span>
                    </div>
                `;
            } else {
                feedHtml += `
                    <div class="feed-item">
                        <div class="feed-info">
                            <div class="feed-avatar unknown">?</div>
                            <div>
                                <div class="feed-name">Unknown Face</div>
                                <div class="feed-meta">No profile matched &bull; ${face.accuracy}% match</div>
                            </div>
                        </div>
                        <span class="feed-badge unknown">Unregistered</span>
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
                    miniAttendanceList.innerHTML = '<p class="text-muted text-center py-4">No attendance marked yet today</p>';
                    return;
                }

                let html = '';
                // Render last 6 entries in reverse chronological order
                const recent = data.records.slice(-6).reverse();
                recent.forEach(item => {
                    const initials = getInitials(item.Name);
                    html += `
                        <div class="mini-log-item">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <div class="row-avatar" style="width: 28px; height: 28px; font-size: 11px;">${initials}</div>
                                <span><strong>${item.Name}</strong> <span class="text-muted">(${item.Enrollment})</span></span>
                            </div>
                            <span class="status-badge present" style="font-size: 11px; padding: 2px 8px;">
                                <i class="fa-solid fa-clock"></i> ${item.Time}
                            </span>
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
            regWebcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" } 
            });
            regWebcamVideo.srcObject = regWebcamStream;
            regCamPlaceholder.classList.add('hidden');

            btnStartRegCam.innerHTML = '<i class="fa-solid fa-stop"></i> Close Camera';
            btnStartRegCam.className = 'btn btn-secondary';
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
        btnStartRegCam.className = 'btn btn-secondary';
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
                if (i <= 10) {
                    const imgThumb = document.createElement('img');
                    imgThumb.src = frameData;
                    imgThumb.className = 'captured-thumb';
                    capturedThumbnails.appendChild(imgThumb);
                }

                await new Promise(r => setTimeout(r, 120)); // Delay between captures
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
                    }, 400);
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
                    playVerificationChime();
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

    // --- TAB 4: ATTENDANCE LOGS TABLE & LIVE FILTERING ---
    async function fetchAttendanceLogs(dateStr) {
        try {
            tbodyAttendance.innerHTML = '<tr><td colspan="6" class="text-center py-4"><i class="fa-solid fa-spinner fa-spin"></i> Fetching records...</td></tr>';
            const res = await fetch(`/api/attendance?date=${dateStr}`);
            const data = await res.json();

            if (data.success && data.records) {
                cachedAttendanceRecords = data.records;
                renderAttendanceLogsTable(cachedAttendanceRecords);
            }
        } catch (err) {
            tbodyAttendance.innerHTML = '<tr><td colspan="6" class="text-center py-4 text-muted">Failed to load attendance logs</td></tr>';
        }
    }

    function renderAttendanceLogsTable(records) {
        if (logsCountChip) {
            logsCountChip.textContent = `${records.length} Record${records.length === 1 ? '' : 's'}`;
        }

        if (records.length === 0) {
            tbodyAttendance.innerHTML = `<tr><td colspan="6" class="text-center py-4 text-muted"><i class="fa-solid fa-folder-open" style="font-size: 24px; display: block; margin-bottom: 8px;"></i> No attendance entries found for selected criteria</td></tr>`;
            return;
        }

        let html = '';
        records.forEach((row, idx) => {
            const initials = getInitials(row.Name);
            html += `
                <tr>
                    <td><strong class="text-muted">${idx + 1}</strong></td>
                    <td><code style="color: var(--accent-indigo-light); font-weight: 600;">${row.Enrollment}</code></td>
                    <td>
                        <div class="student-row-cell">
                            <div class="row-avatar">${initials}</div>
                            <strong>${row.Name}</strong>
                        </div>
                    </td>
                    <td>${row.Date}</td>
                    <td><span class="text-muted"><i class="fa-regular fa-clock"></i> ${row.Time}</span></td>
                    <td>
                        <span class="status-badge present">
                            <i class="fa-solid fa-check"></i> Present
                        </span>
                    </td>
                </tr>
            `;
        });
        tbodyAttendance.innerHTML = html;
    }

    // Real-Time Search Filter on Attendance Logs
    if (inputSearchLogs) {
        inputSearchLogs.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderAttendanceLogsTable(cachedAttendanceRecords);
                return;
            }
            const filtered = cachedAttendanceRecords.filter(r => 
                String(r.Enrollment).toLowerCase().includes(query) ||
                String(r.Name).toLowerCase().includes(query)
            );
            renderAttendanceLogsTable(filtered);
        });
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

    // --- TAB 5: STUDENT DIRECTORY & LIVE SEARCH ---
    async function fetchStudentDirectory() {
        try {
            tbodyStudents.innerHTML = '<tr><td colspan="5" class="text-center py-4"><i class="fa-solid fa-spinner fa-spin"></i> Fetching registered students...</td></tr>';
            const res = await fetch('/api/students');
            const data = await res.json();

            if (data.success && data.students) {
                cachedStudentDirectory = data.students;
                renderStudentDirectoryTable(cachedStudentDirectory);
            }
        } catch (err) {
            tbodyStudents.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted">Failed to load student directory</td></tr>';
        }
    }

    function renderStudentDirectoryTable(students) {
        if (studentsCountChip) {
            studentsCountChip.textContent = `${students.length} Student${students.length === 1 ? '' : 's'}`;
        }

        if (students.length === 0) {
            tbodyStudents.innerHTML = '<tr><td colspan="5" class="text-center py-4 text-muted"><i class="fa-solid fa-user-slash" style="font-size: 24px; display: block; margin-bottom: 8px;"></i> No students found matching criteria</td></tr>';
            return;
        }

        let html = '';
        students.forEach((s, idx) => {
            const initials = getInitials(s.Name);
            html += `
                <tr>
                    <td><strong class="text-muted">${idx + 1}</strong></td>
                    <td><code style="color: var(--accent-indigo-light); font-weight: 600;">${s.Enrollment}</code></td>
                    <td>
                        <div class="student-row-cell">
                            <div class="row-avatar">${initials}</div>
                            <strong>${s.Name}</strong>
                        </div>
                    </td>
                    <td>${s.Date || 'N/A'}</td>
                    <td><span class="text-muted">${s.Time || 'N/A'}</span></td>
                </tr>
            `;
        });
        tbodyStudents.innerHTML = html;
    }

    // Real-Time Search Filter on Student Directory
    if (inputSearchStudents) {
        inputSearchStudents.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (!query) {
                renderStudentDirectoryTable(cachedStudentDirectory);
                return;
            }
            const filtered = cachedStudentDirectory.filter(s => 
                String(s.Enrollment).toLowerCase().includes(query) ||
                String(s.Name).toLowerCase().includes(query)
            );
            renderStudentDirectoryTable(filtered);
        });
    }
});
