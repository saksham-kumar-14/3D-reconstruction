FROM python:3.9

# Install OpenCV dependencies including OpenGL (libGL)
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /3D-reconstruction

COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

COPY . .

CMD ["python", "python/test_depth.py"]
