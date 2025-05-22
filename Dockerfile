# Start from the official Python image, using the 'slim' version for smaller size
FROM python:3.12-slim

# Set environment variable to prevent Python from writing .pyc files and enable buffering
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install build dependencies and clear cache to keep the image light
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the service account key file
COPY gcpcred.json /src/gcpcred.json

# Set the environment variable for Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS=/src/gcpcred.json
ENV OPENAI_API_KEY=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


# Copy the requirements file into the container
COPY ./src/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./src .

# Expose the port your FastAPI app will run on
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
