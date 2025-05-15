# Use an official Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all necessary files
COPY app.py .
COPY requirements.txt .
COPY model/ model/
COPY templates/ templates/
COPY static/ static/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the Flask default port
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]
