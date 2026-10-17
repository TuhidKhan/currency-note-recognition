// ============================================================
// 1. IMPORT HUGGING FACE GRADIO CLIENT (Official SDK)
// ============================================================
import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

// Hugging Face Space repository ID and optional Token
const SPACE_ID = "tuhidkhan/currency-recognition2-api";
const HF_TOKEN = ""; // Optional: Set your Hugging Face user access token here for increased quota


// ============================================================
// 2. DOCUMENT OBJECT MODEL (DOM) REFERENCES
// ============================================================
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzonePrompt = document.getElementById("dropzonePrompt");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const removeBtn = document.getElementById("removeBtn");

const analyzeBtn = document.getElementById("analyzeBtn");
const btnLabel = document.getElementById("btnLabel");
const btnLoader = document.getElementById("btnLoader");
const systemAlert = document.getElementById("systemAlert");

const resultStateBadge = document.getElementById("resultStateBadge");
const idleState = document.getElementById("idleState");
const resultDetails = document.getElementById("resultDetails");

const countryCodeBadge = document.getElementById("countryCodeBadge");
const currencyTitle = document.getElementById("currencyTitle");
const denominationText = document.getElementById("denominationText");
const confidenceValue = document.getElementById("confidenceValue");
const progressIndicator = document.getElementById("progressIndicator");
const matchesList = document.getElementById("matchesList");

let currentFile = null;
let gradioClient = null;


// ============================================================
// 3. FILE SELECTION & DRAG-AND-DROP HANDLERS
// ============================================================
dropzone.addEventListener("click", (event) => {
  if (event.target !== removeBtn) {
    fileInput.click();
  }
});

fileInput.addEventListener("change", (event) => {
  if (event.target.files && event.target.files[0]) {
    loadImage(event.target.files[0]);
  }
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  if (event.dataTransfer.files && event.dataTransfer.files[0]) {
    loadImage(event.dataTransfer.files[0]);
  }
});

function loadImage(file) {
  if (!file.type.startsWith("image/")) {
    alert("Invalid file format. Please upload a JPG or PNG image.");
    return;
  }

  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzonePrompt.style.display = "none";
    previewContainer.style.display = "flex";
    analyzeBtn.disabled = false;

    idleState.style.display = "block";
    resultDetails.style.display = "none";
    systemAlert.style.display = "none";
    resultStateBadge.innerText = "Awaiting Execution";
    resultStateBadge.className = "panel-status-idle";
  };
  reader.readAsDataURL(file);
}

removeBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  currentFile = null;
  fileInput.value = "";
  imagePreview.src = "";
  previewContainer.style.display = "none";
  dropzonePrompt.style.display = "block";
  analyzeBtn.disabled = true;

  idleState.style.display = "block";
  resultDetails.style.display = "none";
  systemAlert.style.display = "none";
  resultStateBadge.innerText = "Awaiting Input";
  resultStateBadge.className = "panel-status-idle";
});


// ============================================================
// 4. API INFERENCE VIA HUGGING FACE GRADIO CLIENT
// ============================================================
analyzeBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  setExecutionState(true);
  systemAlert.style.display = "block";
  systemAlert.style.backgroundColor = "#EFF6FF";
  systemAlert.style.color = "#1D4ED8";
  systemAlert.innerText = "Connecting to Hugging Face ZeroGPU model...";

  try {
    // 1. Connect to Space using official Gradio SDK
    if (!gradioClient) {
      const connectOptions = HF_TOKEN ? { hf_token: HF_TOKEN } : {};
      gradioClient = await Client.connect(SPACE_ID, connectOptions);
    }

    systemAlert.innerText = "Classifying currency note with ResNet-50...";

    // 2. Send image directly to the Gradio endpoint
    const response = await gradioClient.predict("/gradio_predict", {
      img: currentFile
    });

    console.log("Prediction output from Hugging Face:", response);

    // 3. Process the output
    // Gradio Label component returns: { label: "PKR_1000", confidences: [ { label, confidence }, ... ] }
    const outputData = response.data[0];
    displayClassification(outputData);

    systemAlert.style.display = "none";

  } catch (error) {
    console.error("Inference Error:", error);
    systemAlert.style.display = "block";
    systemAlert.style.backgroundColor = "#FEF2F2";
    systemAlert.style.color = "#DC2626";
    systemAlert.innerText = `Error: ${error.message || "Unable to reach backend model."}`;
    resultStateBadge.innerText = "Failed";
    resultStateBadge.className = "panel-status-idle";
  } finally {
    setExecutionState(false);
  }
});


// ============================================================
// 5. HELPER FUNCTIONS
// ============================================================
function setExecutionState(isExecuting) {
  if (isExecuting) {
    analyzeBtn.disabled = true;
    btnLoader.style.display = "inline-block";
    btnLabel.innerText = "Classifying...";
  } else {
    analyzeBtn.disabled = false;
    btnLoader.style.display = "none";
    btnLabel.innerText = "Run Classification";
  }
}

function displayClassification(data) {
  // Support both object with label/confidences and direct dictionary
  let topLabel = "";
  let confidencesList = [];

  if (data.label && data.confidences) {
    topLabel = data.label;
    confidencesList = data.confidences;
  } else if (typeof data === "object") {
    // Dictionary mapping class to confidence
    const sorted = Object.entries(data).sort((a, b) => b[1] - a[1]);
    topLabel = sorted[0][0];
    confidencesList = sorted.map(([k, v]) => ({ label: k, confidence: v }));
  }

  // Check if image is flagged as non-banknote or unrecognized object
  if (topLabel.startsWith("Not a Banknote")) {
    countryCodeBadge.innerText = "UNRECOGNIZED";
    countryCodeBadge.style.backgroundColor = "#FEE2E2";
    countryCodeBadge.style.color = "#DC2626";
    countryCodeBadge.style.borderColor = "#FCA5A5";

    currencyTitle.innerText = "Not a Supported Banknote";
    denominationText.innerText = "Invalid Image";
    denominationText.style.color = "#DC2626";

    confidenceValue.innerText = "Low Confidence";
    progressIndicator.style.width = "10%";
    progressIndicator.style.backgroundColor = "#DC2626";

    systemAlert.style.display = "block";
    systemAlert.style.backgroundColor = "#FEF2F2";
    systemAlert.style.color = "#DC2626";
    systemAlert.innerText = "Unrecognized Object: The model confidence is too low. Please upload a clear photo of an Afghan Afghani or Pakistani Rupee banknote.";

    matchesList.innerHTML = `<div class="match-row" style="color: #64748B;"><span>No confident banknote pattern detected</span></div>`;

    resultStateBadge.innerText = "Rejected";
    resultStateBadge.className = "panel-status-idle";
    idleState.style.display = "none";
    resultDetails.style.display = "block";
    return;
  }

  // Restore standard styling for valid banknotes
  countryCodeBadge.style.backgroundColor = "#DCFCE7";
  countryCodeBadge.style.color = "#059669";
  countryCodeBadge.style.borderColor = "#BBF7D0";
  denominationText.style.color = "#059669";
  progressIndicator.style.backgroundColor = "#059669";

  // Parse Class name (e.g. "PKR_1000" -> currency: "PKR", denom: "1000")
  const parts = topLabel.split("_");
  const currency = parts[0] || "Unknown";
  const denomination = parts[1] || "";
  const isPKR = currency === "PKR";

  // Country badge & title
  countryCodeBadge.innerText = isPKR ? "PKR" : "AFN";
  currencyTitle.innerText = isPKR ? "Pakistani Rupee" : "Afghan Afghani";

  // Denomination value
  denominationText.innerText = `${denomination} ${currency}`;

  // Confidence percentage
  const bestConfidence = confidencesList[0] ? confidencesList[0].confidence : 1.0;
  const confPercent = Math.round(bestConfidence * 100 * 10) / 10;
  confidenceValue.innerText = `${confPercent}%`;
  progressIndicator.style.width = `${Math.min(confPercent, 100)}%`;

  // Matches list
  matchesList.innerHTML = "";
  confidencesList.slice(0, 3).forEach((item) => {
    const row = document.createElement("div");
    row.className = "match-row";
    const cleanName = item.label.replace("_", " ");
    const percent = Math.round(item.confidence * 100 * 10) / 10;
    row.innerHTML = `
      <span>${cleanName}</span>
      <span class="match-score">${percent}%</span>
    `;
    matchesList.appendChild(row);
  });

  resultStateBadge.innerText = "Classified";
  resultStateBadge.className = "panel-status";

  idleState.style.display = "none";
  resultDetails.style.display = "block";
}
