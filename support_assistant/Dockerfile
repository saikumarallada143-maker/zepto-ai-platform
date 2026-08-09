# Task 6: locally buildable/runnable Dockerfile for the FastAPI /ask service.
# Uses python:3.11-slim (not the newest Python) deliberately -- some of this
# module's dependencies (torch, chromadb) are far more likely to have
# prebuilt wheels for a mainstream Python version than for a very new one.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY src/ ./src/
COPY docs/ ./docs/

# MOCK_LLM defaults to mock mode (graded baseline) unless overridden at
# `docker run` time with -e MOCK_LLM=0 -e GROQ_API_KEY=... (optional extension).
ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
