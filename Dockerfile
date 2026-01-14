FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for ODBC
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    odbc-postgresql \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY model/ ./model/

# Set Python path
ENV PYTHONPATH=/app

# Run the processor
CMD ["python", "src/processor.py"]
