# 🏋️‍♂️ Trainix Python API — AI Fitness & Nutrition Microservice

This is the backend microservice for Trainix, providing AI-powered fitness and nutrition analysis via REST API.  
It connects to the main Trainix platform, analyzes user photos, generates personalized plans, and integrates with AWS S3 and OpenAI.

---

## 🚀 Live API Demo

> ⚡️ Free Render server — cold start up to 3 minutes  
[👉 View API Demo *Swagger*](https://trainix-python-api.onrender.com/docs)

---

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **AI/ML:** Mediapipe, OpenAI API
- **Storage:** AWS S3
- **Image Processing:** OpenCV, NumPy
- **DevOps:** Docker, Render
- **Other:** Pydantic, Boto3, Requests

---

## 📦 Features

- AI photo analysis for body metrics
- Fitness plan generation (OpenAI GPT)
- Nutrition plan generation (OpenAI GPT)
- Body measurements calculation
- Photo upload & storage on AWS S3
- CORS support for frontend integration

---

## 📌 API Endpoints

- `POST /api/photo-analyze` — Analyze user photo and return body metrics
- `POST /api/fitnessPlan` — Generate fitness plan - lazy 
- `POST /api/fitnessPlan/day` — Generate detailed plan for a specific day
- `POST /api/nutrition` — Generate daily nutrition plan day

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/gaykun1/Trainix_Project.git
cd Trainix_Project/python-api
pip install --upgrade pip setuptools wheel 
pip install -r requirements.txt
```

### 2. Environment Variables

- Copy `.env.example` to `.env` 
- Fill in your AWS keys, OpenAI API, etc.

### 3. Run Development Server

```bash
uvicorn main:app --reload
```

---

## 🐳 Docker

```bash
cd backend
docker build -t trainix-backend-python .
docker run -p 5200:5200 trainix-backend-python
```

---

## 📄 License

This project is licensed under the [GNU GPL v3](LICENSE).

---

## 💡 Credits

Made with ❤️ by [gaykun1](https://github.com/gaykun1)
