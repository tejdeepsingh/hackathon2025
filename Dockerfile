# 1. Use official Python 3.12 slim image
FROM python:3.12-slim

# 2. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements file
COPY requirements.txt .

# 5. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all app files
COPY . .

# 7. Expose the port Flask runs on
EXPOSE 5000

# 8. Set environment variables (optional but recommended)
ENV FLASK_APP=tejdeep/main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# 9. Run the Flask app
CMD ["flask", "run"]