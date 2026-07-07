FROM python:3.11-slim-bookworm

# Install build dependencies for dlib and opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Pre-install numpy to ensure compilation flags match
RUN pip install --no-cache-dir numpy==1.26.3
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
