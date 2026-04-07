FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# /app/invoice_agent/ is now a proper Python package
# imports like "from invoice_agent.models import ..." resolve correctly
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:7860/health'); r.raise_for_status()" || exit 1

CMD ["uvicorn", "invoice_agent.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
