# Use the official Python image from Docker Hub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install system dependencies required for OpenCV (if needed)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

# Copy the requirements.txt file from the root directory to the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "code.deployment.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
