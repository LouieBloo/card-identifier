# Use the official PyTorch image from Docker Hub
#FROM pytorch/pytorch:2.0.1-cpu
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., OpenCV, wget)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN df -h

RUN pip install --no-cache-dir -r requirements.txt

# RUN git config --global http.postBuffer 524288000 && \
#     git clone --depth 1 https://github.com/ultralytics/yolov5.git
# RUN pip install --no-cache-dir -r yolov5/requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "classify:app", "--host", "0.0.0.0", "--port", "8080"]
