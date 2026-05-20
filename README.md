# HemoLens

Non-invasive hemoglobin estimation from **eye**, **nail**, and **palm** images. FastAPI backend + Expo mobile app.

Repository: [github.com/rexlinw/HemoLens](https://github.com/rexlinw/HemoLens)

## Project structure

```
├── backend/                 # FastAPI + ML
│   ├── app.py               # API server
│   ├── multimodal.py        # Feature fusion & prediction
│   ├── image_validator.py   # Reject invalid/random images
│   ├── feature_extraction.py
│   ├── nail_feature_extraction.py
│   ├── palm_feature_extraction.py
│   ├── eye_detector.py
│   ├── preprocessing.py
│   ├── train.py             # Train multimodal model
│   ├── requirements.txt
│   └── models/              # Production .pkl artifacts
├── mobile/                  # Expo React Native app
│   ├── App.js
│   ├── RealtimeCamera.js
│   └── config.js
├── data/                    # Local datasets (not in git)
│   ├── eyes/
│   │   ├── india/           # Subject folders + India.xlsx
│   │   └── italy/           # Subject folders + Italy.xlsx
│   ├── nails/
│   │   ├── ghana/Fingernails/
│   │   └── standard/Finger_Nails/
│   └── palms/
│       ├── anemic/
│       └── non_anemic/
├── Dockerfile
├── render.yaml
└── README.md
```

## Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

API: `http://localhost:8000`

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health + model metrics |
| `POST /predict` | Single eye image |
| `POST /predict/multimodal` | `eye_file`, `nail_file`, `palm_file` (≥1 required) |

## Mobile

```bash
cd mobile
npm install
npx expo start
```

Local backend:

```bash
EXPO_PUBLIC_API_URL=http://YOUR_LAN_IP:8000 npx expo start
```

## Train model

Place datasets under `data/` as shown above, then:

```bash
cd backend
python train.py
```

Writes `models/hemolens_multimodal.pkl`, `multimodal_scaler.pkl`, `multimodal_config.json`.

## Model performance

See `backend/models/multimodal_config.json` for current metrics (multimodal eye+nail+palm: **R² ≈ 0.85**, **MAE ≈ 0.58 g/dL** on held-out test split).

## Deploy (Render)

1. Connect [rexlinw/HemoLens](https://github.com/rexlinw/HemoLens) to Render
2. Root directory: repository root
3. Use `render.yaml` blueprint or:
   - Build: `pip install -r backend/requirements.txt`
   - Start: `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT`

## WHO reference (g/dL)

| Status | Range |
|--------|--------|
| Low | &lt; 12.0 |
| Borderline | 12.0 – 13.5 |
| Normal | 13.5 – 17.5 |
| High | &gt; 17.5 |

Estimate only — not a medical diagnosis.
