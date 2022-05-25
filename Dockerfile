FROM openjdk:8
COPY --from=python:3.8.13 / /

# user and ssh access
RUN apt update && apt install openssh-server sudo -y
RUN useradd -rm -s /bin/bash -g root -G sudo -u 1000 thoth
RUN echo 'thoth:thoth' | chpasswd
RUN service ssh start
EXPOSE 22
WORKDIR /home/thoth

## requirements
RUN pip install --upgrade pip
COPY requirements.dev.txt .
RUN pip install -r requirements.dev.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

## setup environment
ENV SPARK_VERSION 3.0
RUN env | grep _ >> /etc/environment

## setup package
COPY . .
#RUN pip install .
RUN python -c "import thoth"

## Entrypoint
CMD ["/usr/sbin/sshd","-D"]
