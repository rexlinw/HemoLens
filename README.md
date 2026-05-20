# HemoLens

Non-invasive hemoglobin estimation from eye images using machine learning. FastAPI backend with React Native (Expo) mobile app.

## Structure

```
├── backend/                 # FastAPI API
│   ├── app.py              # Main application
│   ├── eye_detector.py     # Eye detection & health classification
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── requirements.txt
│   └── models/             # Trained models (.pkl)
├── mobile/                  # React Native + Expo app
│   ├── App.js
│   ├── RealtimeCamera.js
│   ├── config.js
│   ├── package.json
│   └── app.json
├── Dockerfile
├── render.yaml
└── README.md
```

## Backend

### Setup

```bash
cd backend
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Server: `http://localhost:8000`

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/info` | Model info |
| POST | `/predict` | Single eye image |
| POST | `/predict/multimodal` | Eye, nail, and/or palm images (`eye_file`, `nail_file`, `palm_file`) |
| POST | `/predict/batch` | Batch eye predictions |

### Train multimodal model

```bash
cd backend
python train_multimodal_production.py
```

## Mobile

### Setup

```bash
cd mobile
npm install
```

### Run

```bash
npx expo start
```

### Local backend

```bash
EXPO_PUBLIC_API_URL=http://YOUR_IP:8000 npx expo start
```

## Deployment

### Render

1. Connect repo to [Render](https://render.com)
2. Create Web Service from `render.yaml`
3. Set Root Directory to repository root
4. Deploy

### Docker

```bash
docker build -t hemolens .
docker run -p 8080:8080 hemolens
```

## Model

- **Algorithm**: Ridge Regression (46 features)
- **Performance (multimodal eye+nail+palm)**: R² ≈ 0.85, MAE ≈ 0.58 g/dL (see `backend/models/multimodal_config.json`)
- **Eye-only fallback**: R² ≈ 0.63, MAE ≈ 0.96 g/dL
- **Features**: RGB, LAB, HSV, YCrCb, statistical, edge, contrast, histogram

## WHO Guidelines

| Status | Range (g/dL) |
|--------|--------------|
| Low | < 12.0 |
| Borderline | 12.0–13.5 |
| Safe | 13.5–17.5 |
| High | > 17.5 |
