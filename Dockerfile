# Use a base image with Python and CUDA (if applicable)
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry && poetry install --no-root

# Copy the application code
COPY src /app/src

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI application
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
