FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install numpy matplotlib scipy Pillow scikit-image opencv-python-headless
CMD ["python3", "watermark_qim.py"]
