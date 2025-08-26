# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies (includes g++, git, ffmpeg, and libgl)
RUN apt-get update && apt-get install -y \
    g++ \
    git \
    ffmpeg \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download models
COPY download_model.sh .
RUN chmod +x download_model.sh && ./download_model.sh

# Copy source code
COPY . .

# Expose app port
EXPOSE 5000

# Start app
CMD ["python", "app.py"]
