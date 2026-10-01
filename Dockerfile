# API container (serving only, no torch). NOTE: written but not built in the authoring environment.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY pyproject.toml .
COPY src ./src
COPY artifacts ./artifacts
COPY reports ./reports
COPY data ./data
RUN pip install --no-cache-dir --no-deps .
EXPOSE 8000
CMD ["uvicorn", "pm.api:app", "--host", "0.0.0.0", "--port", "8000"]
