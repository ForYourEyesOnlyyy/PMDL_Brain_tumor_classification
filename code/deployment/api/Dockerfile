# Dockerfile for FastAPI

# Use the official Python image from Docker Hub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements.txt file from the root directory to the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI code to the container
ENV UVICORN_CMD="uvicorn code.deployment.api.api:app --reload"

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "code.deployment.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
