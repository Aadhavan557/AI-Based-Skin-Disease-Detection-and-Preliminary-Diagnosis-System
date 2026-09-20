// ─────────────────────────────────────────────────────────
//  DermaAI — Image Upload & Analysis Logic
// ─────────────────────────────────────────────────────────

let selectedFile = null;
let cameraStream = null;

document.addEventListener("DOMContentLoaded", () => {
  if (!Auth.requireAuth()) return;

  const user = Auth.getUser();
  if (user) {
    document.querySelectorAll(".user-display-name").forEach(el => el.textContent = user.username);
    document.querySelectorAll(".user-avatar-initial").forEach(el => el.textContent = user.username.charAt(0).toUpperCase());
  }

  // Init dropzone
  const dropZone  = document.getElementById("drop-zone");
  const fileInput  = document.getElementById("file-input");
  const previewSection = document.getElementById("preview-section");
  const uploadSection  = document.getElementById("upload-section");
  const analyzeBtn     = document.getElementById("analyze-btn");
  const previewImg     = document.getElementById("preview-img");

  // Drop events
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault(); dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
  });

  // Handle camera if hash
  if (window.location.hash === "#camera") {
    setTimeout(startCamera, 400);
  }
});

function handleFileSelect(file) {
  const allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showToast("Please upload a JPG, JPEG, or PNG image.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast("Image must be smaller than 10MB.", "error");
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById("preview-img").src = e.target.result;
    document.getElementById("upload-section").style.display  = "none";
    document.getElementById("preview-section").style.display = "flex";
  };
  reader.readAsDataURL(file);
}

function removeImage() {
  selectedFile = null;
  document.getElementById("file-input").value = "";
  document.getElementById("upload-section").style.display  = "block";
  document.getElementById("preview-section").style.display = "none";
  stopCamera();
}

function changeImage() {
  document.getElementById("file-input").click();
}

async function analyzeImage() {
  if (!selectedFile) { showToast("Please select an image first.", "warning"); return; }

  const analyzeBtn = document.getElementById("analyze-btn");
  const scanningSection = document.getElementById("scanning-section");
  const previewSection  = document.getElementById("preview-section");

  // Show scanning state
  previewSection.style.display  = "none";
  scanningSection.style.display = "flex";
  setButtonLoading(analyzeBtn, true, "Analyzing…");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await API.postForm(CONFIG.ENDPOINTS.PREDICT, formData);
    const data = await res.json();

    if (res.ok) {
      // Save result and navigate
      API.saveToHistory(data);
      sessionStorage.setItem(CONFIG.RESULT_KEY, JSON.stringify(data));
      showToast("Analysis complete!", "success");
      setTimeout(() => window.location.href = "result.html", 600);
    } else {
      throw new Error(data.detail || "Analysis failed.");
    }
  } catch (err) {
    showToast(err.message || "Analysis failed. Please try again.", "error");
    previewSection.style.display  = "flex";
    scanningSection.style.display = "none";
  } finally {
    setButtonLoading(analyzeBtn, false);
  }
}

// Camera functionality
async function startCamera() {
  const cameraSection = document.getElementById("camera-section");
  const uploadSection = document.getElementById("upload-section");
  const video = document.getElementById("camera-video");

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = cameraStream;
    uploadSection.style.display  = "none";
    cameraSection.style.display  = "flex";
  } catch (err) {
    showToast("Camera access denied or not available. Please upload a file instead.", "warning");
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  const cameraSection = document.getElementById("camera-section");
  const uploadSection = document.getElementById("upload-section");
  if (cameraSection) cameraSection.style.display = "none";
  if (uploadSection) uploadSection.style.display  = "block";
}

function capturePhoto() {
  const video  = document.getElementById("camera-video");
  const canvas = document.createElement("canvas");
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    if (blob) {
      const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
      handleFileSelect(file);
      stopCamera();
    }
  }, "image/jpeg", 0.92);
}
