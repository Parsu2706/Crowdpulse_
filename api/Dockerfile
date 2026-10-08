FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt api/requirements.txt
# CPU-only torch keeps the image ~2GB smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r api/requirements.txt \
    && python -m spacy download en_core_web_sm

COPY api ./api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
