FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=300 \
    -i https://pypi.org/simple \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "chatbot_fast:app", "--host", "0.0.0.0", "--port", "8000"]