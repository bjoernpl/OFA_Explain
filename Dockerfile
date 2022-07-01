FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt-get -y install tzdata
RUN apt-get install ffmpeg libsm6 libxext6 webp libwebp-dev -y

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY fairseq ./fairseq
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python==4.5.5.64

COPY . .

EXPOSE 8080
ENTRYPOINT [ "python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
