#!/bin/bash
echo "✅ Model downloaded and extracted"

mkdir -p .insightface/models/buffalo_l

echo "📥 Downloading model files..."

gdown --id 1lqb107Kp3wDBoiTI0Zmltt0QM_qJ5-Nm -O .insightface/models/buffalo_l/config.yaml
gdown --id 1VHwR1nePjt0nOQXZR5SVPhioHhN91dcJ -O .insightface/models/buffalo_l/det_10g.onnx
gdown --id 185NNgxPoe5XDkCMyWaLkyJeCGHBedo_f -O .insightface/models/buffalo_l/w600k_r50.onnx
gdown --id 19XQbQzHoMOMqbgv63FBvvdab19twQ7xx -O .insightface/models/buffalo_l/1k3d68.onnx
gdown --id 1x_oht716_lkGsM75gHNd9XN2bNpsVynh -O .insightface/models/buffalo_l/2d106det.onnx
gdown --id 12xiMN1lVau9ZSArPsfjNewvpTHx27HJK -O .insightface/models/buffalo_l/genderage.onnx
