FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install numpy matplotlib scipy Pillow

CMD ["python3", "watermark_qim.py"]
