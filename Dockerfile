FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install numpy matplotlib scipy Pillow scikit-image opencv-python-headless flask
EXPOSE 80
CMD ["python3", "app.py"]
