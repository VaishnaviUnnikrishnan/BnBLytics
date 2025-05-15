# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code
COPY train.py .
COPY AB_NYC_2019.csv .
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run training script
CMD ["python", "train.py"]
