# Use a base image with Python and CUDA (if applicable)
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY requirements.txt /app/requirements.txt

# Install Poetry and dependencies
RUN pip install -r requirements.txt

# Copy the application code
COPY src /app/src

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]