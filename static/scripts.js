// Countdown Timer for Admin Dashboard
function fetchRemainingTime() {
    fetch('/remaining_time')
        .then(response => response.json())
        .then(data => {
            const remainingTimeElement = document.getElementById('remaining-time');
            if (data.time) {
                remainingTimeElement.textContent = data.time;
            } else {
                remainingTimeElement.textContent = "No ongoing lecture";
            }
        });
}

// Update Remaining Time Every Second
if (document.getElementById('remaining-time')) {
    setInterval(fetchRemainingTime, 1000);
}

// Live Face Detection for User Dashboard
if (document.getElementById('webcam')) {
    const video = document.getElementById('webcam');
    const verificationStatus = document.getElementById('verification-status');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Start Webcam Feed
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
    });

    async function processFrame() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
        const formData = new FormData();
        formData.append('image', blob);

        const response = await fetch('/live_verify', { method: 'POST', body: formData });
        const data = await response.json();

        // Draw Bounding Box and Update Status
        if (data.rect) {
            const [x1, y1, x2, y2] = data.rect;
            ctx.strokeStyle = 'blue';
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = 'white';
            ctx.font = '14px Arial';
            ctx.fillText(`${data.name} (${data.confidence}%)`, x1, y1 - 10);
        }

        verificationStatus.textContent = `Status: ${data.verified ? 'Verified' : 'Unverified'}`;
        requestAnimationFrame(processFrame);
    }

    video.addEventListener('loadeddata', () => {
        processFrame();
    });
}

// Verify Attendance Button for User Dashboard
if (document.getElementById('verify-ip')) {
    const verifyButton = document.getElementById('verify-ip');
    const verificationStatus = document.getElementById('verification-status');

    verifyButton.addEventListener('click', async () => {
        const response = await fetch('/verify_ip', { method: 'POST' });
        const data = await response.json();
        verificationStatus.textContent = `Status: ${data.status} (Your IP: ${data.user_ip})`;
    });
}
