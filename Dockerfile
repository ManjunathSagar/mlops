FROM python:3.10-slim

# Set environment
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt && \
    python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab averaged_perceptron_tagger_eng

# Run FastAPI app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]