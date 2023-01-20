FROM ubuntu:20.04

WORKDIR /flask-api-object-detection

COPY /flask-api-object-detection/requirements.txt .
RUN apt-get update && \
    apt-get install -y python3-pip 
RUN apt install -y libusb-1.0-0-dev
RUN apt install -y libportaudio2
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

COPY /flask-api-object-detection/ .


EXPOSE 5000

CMD ["python3", "-m", "flask", "--app", "/flask-api-object-detection/detect1.py", "run","--host=0.0.0.0"]
