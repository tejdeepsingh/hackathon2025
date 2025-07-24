#!/bin/bash

# ===== Configuration =====
IMAGE_NAME="asterisk"
CONTAINER_NAME="asterisk"
RECORDINGS_DIR="./recordings"
LOGS_DIR="./logs"

# ===== Build Docker Image =====
echo "🚧 Building Docker image..."
docker build -t $IMAGE_NAME .

# ===== Create Local Directories =====
echo "📁 Ensuring recordings and logs directories exist..."
mkdir -p "$RECORDINGS_DIR" "$LOGS_DIR"

# ===== Set Correct Permissions =====
echo "🔐 Fixing directory permissions (UID 999 for asterisk user)..."
sudo chown -R 999:999 "$RECORDINGS_DIR" "$LOGS_DIR"

# ===== Run Docker Container =====
echo "🚀 Running Docker container..."
docker run -it --rm \
  --name "$CONTAINER_NAME" \
  -v "$(pwd)/pjsip.conf:/etc/asterisk/pjsip.conf" \
  -v "$(pwd)/extensions.conf:/etc/asterisk/extensions.conf" \
  -v "$(pwd)/cdr_scripts.sh:/usr/local/bin/cdr_scripts.sh" \
  -v "$(pwd)/recordings:/var/spool/asterisk/monitor" \
  -v "$(pwd)/logs:/var/log/asterisk" \
  "$IMAGE_NAME"

