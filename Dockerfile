FROM python:3.10-slim

WORKDIR /app

# creating user 
RUN addgroup myGroup && adduser --ingroup myGroup User

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install --upgrade openai

COPY . .

RUN chown -R User:myGroup /app

USER User

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
