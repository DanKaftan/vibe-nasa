#!/bin/bash

apt-get update

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC 
apt-get -y install tzdata wget

# # ── Base OS packages ─────────────────────────────────────────────────────────
apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir -r requirements.txt --break-system-packages

echo "Done content update"
