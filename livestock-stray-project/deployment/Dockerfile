FROM python:3.10-slim
WORKDIR /app

RUN pip install --no-cache-dir ultralytics fastapi uvicorn python-multipart pillow

COPY models/ models/
COPY deployment/edge_service.py .
COPY deployment/alerts.py .

ENV SCENARIO=livestock-stray
ENV MODEL_PATH=models/best.pt
ENV OOD_THRESHOLD=0.25
ENV OOD_LOG_DIR=/app/ood_queue

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "edge_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
