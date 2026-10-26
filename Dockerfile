FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    OHAM_DATA_DIR=/data/ohamlab-agent-lab

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir .
RUN mkdir -p /data/ohamlab-agent-lab

EXPOSE 7860
CMD ["python", "app.py"]
