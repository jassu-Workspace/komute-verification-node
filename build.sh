#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "🚀 Building Komüte Driver Verification Microservice v2"
echo "=========================================================="

# 1. Upgrade pip, wheel and setuptools
echo "⚙️ Upgrading Python packaging toolchain..."
python -m pip install --upgrade pip setuptools wheel

# 2. Install production dependencies
echo "📦 Installing application dependencies from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# 3. Create persistent/temporary upload directories
echo "📁 Initializing storage directories..."
mkdir -p backend/storage/uploads/drivers \
         backend/storage/uploads/previews \
         backend/uploads

# 4. Verify AI/DNN model assets
echo "🔍 Checking neural network model assets..."
if [ -f "backend/models/face_detection_yunet_2023mar.onnx" ]; then
    echo "✅ Face detection YuNet ONNX model verified at backend/models/face_detection_yunet_2023mar.onnx"
else
    echo "⚠️ YuNet model file missing in backend/models/!"
fi

echo "=========================================================="
echo "✅ Build completed successfully! Microservice ready."
echo "=========================================================="
