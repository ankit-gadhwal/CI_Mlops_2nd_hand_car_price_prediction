# basic image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# copy application code
COPY . /app

# Install dependencies
RUN pip install -r docker_api_req/requirements.txt

# Expose Gradio port
Expose 7860

# Run the application
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
