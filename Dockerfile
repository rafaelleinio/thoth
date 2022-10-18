FROM openjdk:8 as dependencies
COPY --from=python:3.9 / /

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
FROM dependencies as thoth
WORKDIR /app

COPY . /app
RUN pip install /app/.
RUN python -c "import thoth"

## start UI
FROM thoth as ui

CMD streamlit run /app/ui.py
