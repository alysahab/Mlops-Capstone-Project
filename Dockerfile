FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (best practice)
COPY setup.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Then copy the rest
COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

#local
# CMD ["python", "app.py"]


#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]