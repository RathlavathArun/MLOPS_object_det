FROM python:3.10-slim

WORKDIR /app

# System deps (for OpenCV & general builds)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip

# Install CPU PyTorch explicitly + your libs
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi uvicorn opencv-python-headless prometheus-client ultralytics python-multipart

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]