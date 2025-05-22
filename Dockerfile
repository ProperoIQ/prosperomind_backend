# Use an official Python runtime as the base image
FROM python:3.12-slim



# Copy the requirements file into the container
WORKDIR /app/src

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
