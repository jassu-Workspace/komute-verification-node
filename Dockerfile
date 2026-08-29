FROM python:3.11-slim

# Install system dependencies for OpenCV, ONNX Runtime, and health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full application codebase
COPY . .

# Initialize uploads & storage directories
RUN mkdir -p /app/backend/storage/uploads/drivers \
             /app/backend/storage/uploads/previews \
             /app/backend/uploads

# Environment variables (Render dynamically injects PORT)
ENV PORT=8080
ENV HOST=0.0.0.0
ENV ENVIRONMENT=production
ENV PYTHONUNBUFFERED=1

# Expose default port
EXPOSE 8080

# Health check node verification
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Set working directory to backend for clean module resolution
WORKDIR /app/backend

# Launch server dynamically binding to Render's assigned $PORT
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]

