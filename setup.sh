#!/bin/bash

# 1. Install packages from requirements.txt if it exists
REQ_FILE="./requirements.txt"

if [ -f "$REQ_FILE" ]; then
    echo "[1] requirements.txt found. Installing packages..."
    pip install -r "$REQ_FILE"
    if [ $? -ne 0 ]; then
        echo "[!] Package installation failed. Exiting."
        exit 1
    fi
else
    echo "[1] requirements.txt not found. Skipping pip install."
fi

# 2. Download Hugging Face model (.pth) file to ~/.cache
echo "[2] Downloading Hugging Face model file to ~/.cache..."

HF_URL="https://huggingface.co/xingren23/comfyflow-models/resolve/976de8449674de379b02c144d0b3cfa2b61482f2/sams/sam_vit_b_01ec64.pth"
HF_CACHE_DIR="$HOME/.cache"
HF_FILE_NAME="sam_vit_b_01ec64.pth"
HF_DEST_PATH="$HF_CACHE_DIR/$HF_FILE_NAME"

mkdir -p "$HF_CACHE_DIR"

curl -L "$HF_URL" -o "$HF_DEST_PATH"
if [ -f "$HF_DEST_PATH" ]; then
    echo "[2] Model file successfully saved to $HF_DEST_PATH"
else
    echo "[!] Failed to download Hugging Face model file. Exiting."
    exit 1
fi

# 3. Download and extract VisA tar file
echo "[3] Downloading and extracting VisA .tar file..."

# Replace this with the actual download URL of the VisA tar file
VISA_URL="https://example.com/path/to/VisA_20220922.tar"
VISA_TAR_NAME="VisA_20220922.tar"
VISA_TEMP_PATH="/tmp/$VISA_TAR_NAME"

curl -L "$VISA_URL" -o "$VISA_TEMP_PATH"
if [ ! -f "$VISA_TEMP_PATH" ]; then
    echo "[!] Failed to download VisA tar file. Exiting."
    exit 1
fi

# Create target directory and move the tar file
mkdir -p /workspace/data/visa
mv "$VISA_TEMP_PATH" /workspace/data/visa/

# Change to the target directory and extract the tar file
cd /workspace/data/visa || { echo "[!] Failed to change to visa directory."; exit 1; }
tar -xvf "$VISA_TAR_NAME"

# Delete the tar file after extraction
rm "$VISA_TAR_NAME"
echo "[3] VisA tar file extracted and removed."

echo "[✓] All tasks completed successfully."
