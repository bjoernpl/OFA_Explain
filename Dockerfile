FROM python:3.9-buster

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY fairseq ./fairseq
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
ENTRYPOINT [ "python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
