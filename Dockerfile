# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required to build packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source code
COPY . .

# Expose Flask port
EXPOSE 5000

# Final command to run
CMD bash -c "bash download_model.sh && python app.py"
