FROM python:3.10-slim

# Install system dependencies for PyMuPDF, Tesseract OCR, FAISS, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set NLTK data path to a writable directory in /tmp
ENV NLTK_DATA=/tmp/nltk_data
RUN mkdir -p ${NLTK_DATA}

# Download required NLTK data at build time to the specified NLTK_DATA path
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt stopwords

COPY . .

EXPOSE 7860

# Set Hugging Face cache directory to writable location
ENV HF_HOME=/tmp/hf_cache
# Create cache directories with proper permissions
RUN mkdir -p /tmp/hf_cache && chmod 777 /tmp/hf_cache

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
