FROM python:3.11-slim

WORKDIR /app

# tesseract-ocr + poppler-utils enable the OCR fallback for scanned PDFs (optional
# at the Python-package level, but required here for that feature to actually work
# inside the container — see README.md "OCR fallback for scanned PDFs").
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p storage/uploads storage/vector_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
