# Use the official Python 3.12 image as the base
FROM python:3.12.7-slim

# Set the working directory inside the container
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the Poetry files first (to cache dependencies)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-dev --no-interaction --no-ansi

## Set AWS_PROFILE environment variable or mention it at run-time
#ENV AWS_PROFILE=PowerUserAccess-691506669376

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN poetry add rasterio  


# Copy the entire project into the container
COPY . /app

# Specify the entry point or command
CMD ["python", "inference.py"]
