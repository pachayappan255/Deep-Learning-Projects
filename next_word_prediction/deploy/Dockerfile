# Build from the project root:  docker build -f deploy/Dockerfile -t nwp .
FROM python:3.12-slim
WORKDIR /app
COPY deploy/requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt
COPY nwp ./nwp
COPY deploy ./deploy
COPY runs/transformer/model.pt ./runs/transformer/model.pt
ENV PORT=8000
CMD ["sh", "-c", "uvicorn deploy.api:app --host 0.0.0.0 --port ${PORT}"]
