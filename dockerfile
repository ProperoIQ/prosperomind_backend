# Use an official Python runtime as the base image
FROM python:3.9-slim



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