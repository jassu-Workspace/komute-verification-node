# 🚀 Deploying Komüte Driver Verification Service on Render

This guide provides step-by-step instructions to deploy the **Komüte Driver Verification Service v2** on [Render](https://render.com).

---

## 📑 Table of Contents
1. [Architecture & Deployment Features](#architecture--deployment-features)
2. [Option 1: Deploy via Render Blueprint (Recommended - 1 Click)](#option-1-deploy-via-render-blueprint-recommended)
3. [Option 2: Deploy as Native Python Web Service](#option-2-deploy-as-native-python-web-service)
4. [Option 3: Deploy as Docker Web Service](#option-3-deploy-as-docker-web-service)
5. [Health Check Node & Monitoring](#health-check-node--monitoring)
6. [Sending API Requests to Render](#sending-api-requests-to-render)
7. [Environment Variables Reference](#environment-variables-reference)

---

## 🛡️ Architecture & Deployment Features

- **Dynamic Port Binding**: Automatically binds to Render's dynamic `$PORT` environment variable (`0.0.0.0:$PORT`).
- **Cloud OCR & Headless Computer Vision**: Preconfigured with `opencv-python-headless` and RapidOCR ONNX runtime to prevent Linux GUI/X11 library missing errors.
- **Dedicated Health & Diagnostics Node**: Endpoints `/health`, `/ping`, and `/api/v1/health` for zero-downtime health checking.
- **Interactive Web Dashboard**: Accessible directly at `https://<your-render-url>.onrender.com/dashboard` or `/`.

---

## ⚡ Option 1: Deploy via Render Blueprint (Recommended)

The repository includes a ready-to-use [`render.yaml`](render.yaml) blueprint.

1. Push your repository to **GitHub** or **GitLab**.
2. Log in to [Render Dashboard](https://dashboard.render.com).
3. Click **New +** → **Blueprint**.
4. Connect your GitHub/GitLab repository.
5. Render will automatically detect [`render.yaml`](render.yaml) and configure:
   - **Runtime**: Python 3.11.9
   - **Build Command**: `./build.sh`
   - **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
6. Enter your `GEMINI_API_KEY` in the environment variable prompt.
7. Click **Apply**. Render will build and deploy the service.

---

## 🐍 Option 2: Deploy as Native Python Web Service

If setting up manually in the Render dashboard:

1. Click **New +** → **Web Service**.
2. Select your repository.
3. Configure the service settings:
   - **Name**: `komute-driver-verifier`
   - **Language / Runtime**: `Python`
   - **Region**: `Oregon (US West)` or closest to your users
   - **Branch**: `main` (or your active branch)
   - **Build Command**:
     ```bash
     chmod +x ./build.sh && ./build.sh
     ```
   - **Start Command**:
     ```bash
     cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free` or `Starter` (0.5+ CPU, 512MB+ RAM recommended)
4. Under **Advanced Settings**:
   - **Health Check Path**: `/health`
   - Add the following **Environment Variables**:
     - `PYTHON_VERSION` = `3.11.9`
     - `ENVIRONMENT` = `production`
     - `VLM_PROVIDER` = `gemini`
     - `VLM_MODEL` = `gemini-2.5-flash`
     - `GEMINI_API_KEY` = `<YOUR_GOOGLE_GEMINI_API_KEY>`
     - `ENABLE_API_KEY_AUTH` = `false` *(set to `true` if you want API Key protection)*
5. Click **Create Web Service**.

---

## 🐳 Option 3: Deploy as Docker Web Service

For 100% containerized execution using the production [`Dockerfile`](Dockerfile):

1. Click **New +** → **Web Service**.
2. Select your repository.
3. Configure:
   - **Runtime**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.`
4. Under **Advanced Settings**:
   - **Health Check Path**: `/health`
   - Add environment variables (`GEMINI_API_KEY`, `VLM_PROVIDER`, etc.)
5. Click **Create Web Service**.

---

## 💓 Health Check Node & Monitoring

The application includes a built-in health check node.

### 1. Primary Health Check (`GET /health` or `GET /api/v1/health`)
Render polls this endpoint to verify zero-downtime deployments.

**Example Request:**
```bash
curl -i https://your-service.onrender.com/health
```

**Example 200 OK Response:**
```json
{
  "status": "healthy",
  "app": "Komüte Driver Verification Service v2",
  "version": "2.0.0",
  "environment": "production",
  "uptime_seconds": 128.45,
  "uptime_human": "2m 8s",
  "timestamp": "2026-08-29T11:45:00.000000Z",
  "vlm_provider": "gemini",
  "vlm_model": "gemini-2.5-flash",
  "ocr_available": true,
  "face_detector_available": true,
  "checks": {
    "ocr_engine": true,
    "face_detector": true,
    "storage_writable": true,
    "memory_usage_mb": 115.4
  }
}
```

### 2. Lightweight Ping Probe (`GET /ping`)
For load balancers or high-frequency uptime monitors:
```bash
curl https://your-service.onrender.com/ping
```
```json
{
  "status": "ok",
  "ping": "pong",
  "timestamp": "2026-08-29T11:45:00.000000Z",
  "app": "Komüte Driver Verification Service v2",
  "version": "2.0.0"
}
```

---

## 📡 Sending API Requests to Render

Once deployed on Render at `https://your-service.onrender.com`:

### 1. Interactive UI Dashboard
Open your browser and navigate to:
```
https://your-service.onrender.com/dashboard
```

### 2. Interactive Swagger / OpenAPI Docs
```
https://your-service.onrender.com/docs
```

### 3. Full 3-Stage Driver Verification (`POST /api/v1/verify`)
```bash
curl -X POST "https://your-service.onrender.com/api/v1/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_prod_101",
    "driver_id": "drv_prod_101",
    "personal_info": {
      "full_name": "Jaswanth Sri Sai Venkat Dangeti",
      "email_address": "driver@komute.com",
      "mobile_number": "+918500923656",
      "date_of_birth": "2005-12-14",
      "role": "Driver"
    },
    "license_details": {
      "license_number": "DDNPV1331Q",
      "issuing_province": "ON",
      "license_expiry_date": "2030-12-14"
    },
    "vehicle_details": {
      "make": "Hyundai",
      "model": "Venue",
      "year": "2020",
      "color": "Black",
      "plate": "DL 7CQ 1939"
    },
    "images": {
      "selfie_base64": "<BASE64_SELFIE_STRING>",
      "license_image_base64": "<BASE64_DL_STRING>",
      "vehicle_photo_base64": "<BASE64_VEHICLE_STRING>"
    }
  }'
```

### 4. Zero-PII Face Privacy Preview (`POST /api/v1/privacy-crop-preview`)
```bash
curl -X POST "https://your-service.onrender.com/api/v1/privacy-crop-preview" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<BASE64_DL_STRING>",
    "margin_ratio": 0.15
  }'
```

### 5. WebP/AVIF Image Compression (`POST /api/v1/compress`)
```bash
curl -X POST "https://your-service.onrender.com/api/v1/compress" \
  -F "file=@/path/to/image.jpg" \
  -F "compressionPercentage=80" \
  -F "format=webp" \
  --output compressed_image.webp
```

---

## 🔑 Environment Variables Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `PORT` | Integer | `8080` | Render-assigned server listening port. |
| `HOST` | String | `0.0.0.0` | Bind host address. |
| `ENVIRONMENT` | String | `production` | Deployment mode (`production` or `development`). |
| `VLM_PROVIDER` | String | `gemini` | Cloud AI model provider (`gemini`, `anthropic`, `mock`). |
| `VLM_MODEL` | String | `gemini-2.5-flash` | Cloud VLM model identifier. |
| `GEMINI_API_KEY` | String | *(Optional)* | Google AI Gemini API Key. |
| `ENABLE_API_KEY_AUTH` | Boolean | `false` | Set `true` to require `X-API-Key` header on API endpoints. |
| `API_SECRET_KEY` | String | *(Generated)* | Secret API key if authentication is enabled. |
| `RATE_LIMIT_PER_MINUTE` | Integer | `60` | Max requests per minute per IP address. |
| `OMP_NUM_THREADS` | Integer | `2` | CPU thread cap for low latency and smooth concurrency. |
