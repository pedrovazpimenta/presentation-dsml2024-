FROM python:3.12.3-slim
RUN useradd -m app

ADD . /app
RUN chown -R app:app /app/
WORKDIR /app

# opencv dependencies 
RUN apt-get update && apt-get install -y libgl1 libjpeg-dev libglib2.0-0

# tesseract
RUN apt-get install -y tesseract-ocr

RUN python3.12 -m pip install --upgrade pip setuptools wheel
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* && rm -fr /root/.cache

USER app

CMD bash