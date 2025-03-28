# MLflow-FastAPI Model Serving

This project provides a full **ML pipeline** using **MLflow** for model tracking and **FastAPI** for deployment.

## 🚀 Features
- Train and log models using MLflow
- Serve models via FastAPI
- Track experiments and metrics
- REST API for predictions

---

## 🛠 Installation

### 1️⃣ **Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### 2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run MLflow Tracking Server**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host(http://127.0.0.1/)--port 5000
```

---

## 📊 Training and Logging Model
Run the `train.py` script to train and log the model:
```bash
python train.py
```

This will:
- Train a machine learning model
- Log the model and metrics in MLflow

---

## 🔥 Deploy Model with FastAPI
Start the FastAPI server to serve predictions:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints
### 🔹 **Health Check**
```http
GET /
```
Response:
```json
{"message": "ML Model Serving is Running!"}
```

### 🔹 **Predict**
```http
POST /predict
```
#### **Request (JSON)**
```json
{
  "feature1": 1.2,
  "feature2": 3.4
}
```
#### **Response (JSON)**
```json
{
  "prediction": 0.87
}
```

---

## 🏗 Project Structure
```
MLflow-FastAPI-Serving/
├── venv/               # Virtual environment
├── app.py              # FastAPI server
├── train.py            # Training script
├── requirements.txt    # Dependencies        
├── mlruns/             # Stored ML models
└── README.md           # Documentation
```

---
## MLflow Experiment Tracking

Below is a screenshot of the MLflow UI showing experiment runs:

<img width="956" alt="image" src="https://github.com/user-attachments/assets/9fa3cd32-9f70-4d71-8ad9-a089a346f362" />


----

## 💡 Next Steps
- 🔍 Improve model performance
- 📦 Containerize with Docker
- 🚀 Deploy on cloud (AWS/GCP)


