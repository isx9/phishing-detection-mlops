# Install the libraries
    FROM python:3.9-slim AS builder
    WORKDIR /build
    COPY requirements-api.txt .
    RUN pip install --user --no-cache-dir -r requirements-api.txt
    
# The image that actually runs
    FROM python:3.9-slim AS app
    COPY --from=builder /root/.local /root/.local
    ENV PATH=/root/.local/bin:$PATH
    WORKDIR /app
    COPY api.py .
    COPY model/ model/
    EXPOSE 8000
    CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]