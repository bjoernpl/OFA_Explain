FROM python:3.9-buster

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY fairseq ./fairseq
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT [ "python", "-m", "uvicorn", "api:app"]