#!/bin/bash

echo "📁 Creating model directory..."
mkdir -p /root/.insightface/models/buffalo_l
cd /root/.insightface/models/buffalo_l

echo "📥 Downloading model files..."

gdown https://drive.google.com/uc?id=1lqb107Kp3wDBoiTI0Zmltt0QM_qJ5-Nm -O config.yaml
gdown https://drive.google.com/uc?id=1VHwR1nePjt0nOQXZR5SVPhioHhN91dcJ -O det_10g.onnx
gdown https://drive.google.com/uc?id=185NNgxPoe5XDkCMyWaLkyJeCGHBedo_f -O w600k_r50.onnx
gdown https://drive.google.com/uc?id=19XQbQzHoMOMqbgv63FBvvdab19twQ7xx -O 1k3d68.onnx
gdown https://drive.google.com/uc?id=1x_oht716_lkGsM75gHNd9XN2bNpsVynh -O 2d106det.onnx
gdown https://drive.google.com/uc?id=12xiMN1lVau9ZSArPsfjNewvpTHx27HJK -O genderage.onnx

echo "✅ Model downloaded and saved in /root/.insightface/models/buffalo_l"
