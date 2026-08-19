FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[search,redis]'
USER 65532:65532
EXPOSE 8080
CMD ["uvicorn", "energy_agent.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
