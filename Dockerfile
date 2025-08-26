# Use official slim Python 3.10.13 image
FROM python:3.10.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for onnxruntime and opencv
RUN apt-get update && \
    apt-get install -y git curl ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Download the model
RUN chmod +x download_model.sh && bash download_model.sh

# Expose port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]