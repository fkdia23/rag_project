FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python3", "-m", "qabot.py"]