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
| POST | `/predict` | Single image prediction |
| POST | `/predict/batch` | Batch predictions |

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
- **Performance**: R² = 0.6267, MAE = 0.96 g/dL
- **Features**: RGB, LAB, HSV, YCrCb, statistical, edge, contrast, histogram

## WHO Guidelines

| Status | Range (g/dL) |
|--------|--------------|
| Low | < 12.0 |
| Borderline | 12.0–13.5 |
| Safe | 13.5–17.5 |
| High | > 17.5 |
