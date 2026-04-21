FROM python:3.11-slim

WORKDIR /app
<<<<<<< HEAD

COPY . .

RUN pip install numpy matplotlib scipy Pillow

CMD ["python3", "watermark_qim.py"]
=======
EXPOSE 8000
CMD ["python", "watermark_qim.py"]
>>>>>>> 081d427 (9:54 commit)
