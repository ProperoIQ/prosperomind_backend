

# Install the dependencies


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY gcpcred.json /src/gcpcred.json

# Set the environment variable for Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/src/gcpcred.json
ENV OPENAI_API_KEY=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


COPY ./src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]


