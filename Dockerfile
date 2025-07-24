# 1. Use official Python 3.11 slim image
FROM python:3.12-slim
# 2. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   ffmpeg \
   && rm -rf /var/lib/apt/lists/*
# 3. Set working directory
WORKDIR /app
# 4. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt
# 5. Copy all app files
COPY . .
# 6. Expose the port Streamlit runs on
EXPOSE 8080
# 7. Launch your Streamlit app
CMD ["streamlit", "run", main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]