# DermaAI - AI-Based Skin Disease Detection and Preliminary Diagnosis System

DermaAI is a full-stack web application designed to assist in the preliminary detection and analysis of various skin conditions using advanced Deep Learning. It allows users to upload images of skin concerns and receive instant, AI-assisted predictions, along with visual GradCAM heatmaps and detailed disease information.

## 🌟 Features

- **AI-Powered Analysis**: Utilizes a PyTorch-based EfficientNet deep learning model for high-accuracy skin disease classification.
- **Visual Explainability (GradCAM)**: Generates heatmaps highlighting the exact regions of the image that the AI focused on to make its prediction.
- **User Authentication**: Secure JWT-based login and registration system.
- **Prediction History**: Authenticated users can view their past scans and reports.
- **Modern UI/UX**: Clean, responsive, and intuitive frontend built with vanilla HTML/CSS/JS.

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: FastAPI (Python)
- **AI/ML**: PyTorch, Torchvision, OpenCV, Matplotlib
- **Database**: MongoDB (Local or Atlas via Motor async driver)

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- MongoDB instance (Local or Atlas)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd "AI-Based Skin Disease Detection and Preliminary Diagnosis System"
```

### 2. Backend Setup
Configure your environment variables and start the FastAPI server:

```bash
# Set up a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install backend dependencies
pip install -r backend/requirements.txt

# Set up environment variables
# 1. Rename `.env.example` to `.env`
# 2. Add your MongoDB Connection String and JWT Secret Key to the `.env` file

# Start the Backend Server
uvicorn backend.main:app --reload
```
The backend API will run at `http://127.0.0.1:8000`.

### 3. Frontend Setup
Open a new terminal window to serve the static frontend files:

```bash
# Navigate to the frontend directory
cd frontend

# Start a simple HTTP server
python -m http.server 3000
```
Open your browser and navigate to `http://localhost:3000` to use the application.

## 📁 Project Structure

```text
├── backend/                  # FastAPI backend application
│   ├── main.py               # Application entry point
│   ├── models/               # Pydantic schemas and DB models
│   ├── routes/               # API endpoints (auth, predictions, etc.)
│   ├── services/             # Core business logic (predict_service, etc.)
│   └── requirements.txt      # Python dependencies
├── frontend/                 # Frontend UI
│   ├── index.html            # Landing page
│   ├── analyze.html          # Upload and analysis portal
│   ├── css/                  # Stylesheets
│   └── js/                   # Vanilla JS logic
├── dataset/                  # (Git ignored) Raw training data
├── models/                   # PyTorch Model training and architecture scripts
├── preprocessing/            # Scripts for data cleaning and augmentation
├── .env.example              # Example environment variables
└── .gitignore                # Git ignore rules
```

## ⚠️ Disclaimer
**This application is for informational and educational purposes only.** The AI predictions provided by DermaAI do not constitute professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider or dermatologist with any questions you may have regarding a medical condition.
