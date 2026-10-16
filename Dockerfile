FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONOPS_MODEL=/models/person.onnx \
    VISIONOPS_PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 visionops

WORKDIR /app
COPY pyproject.toml ./
COPY visionops ./visionops
RUN pip install --no-cache-dir .

USER visionops
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

CMD ["sh", "-c", "python -m visionops serve --model \"$VISIONOPS_MODEL\" --host 0.0.0.0 --port \"$VISIONOPS_PORT\""]
