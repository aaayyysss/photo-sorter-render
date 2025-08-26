#!/bin/bash
set -e

echo "📁 Creating model directory..."
mkdir -p /root/.insightface/models/buffalo_l
cd /root/.insightface/models/buffalo_l

echo "📥 Downloading model files..."

gdown https://drive.google.com/uc?id=1lqb107Kp3wDBoiTI0Zmltt0QM_qJ5-Nm
gdown https://drive.google.com/uc?id=1VHwR1nePjt0nOQXZR5SVPhioHhN91dcJ
gdown https://drive.google.com/uc?id=185NNgxPoe5XDkCMyWaLkyJeCGHBedo_f
gdown https://drive.google.com/uc?id=19XQbQzHoMOMqbgv63FBvvdab19twQ7xx
gdown https://drive.google.com/uc?id=1x_oht716_lkGsM75gHNd9XN2bNpsVynh
gdown https://drive.google.com/uc?id=12xiMN1lVau9ZSArPsfjNewvpTHx27HJK

echo "✅ Model downloaded and saved in /root/.insightface/models/buffalo_l"
