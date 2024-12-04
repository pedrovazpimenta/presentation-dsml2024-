FROM python:3.12.3-slim

ADD . /bench
WORKDIR /bench

RUN python3.12 -m pip install --upgrade pip setuptools wheel
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

CMD bash