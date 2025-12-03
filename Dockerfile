FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

ENV POETRY_VERSION=1.7.1
ENV POETRY_VENV=/app/.venv
ENV no_proxy=localhost
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app
WORKDIR /app


# Installing system deps for GDAL
RUN apt-get update
RUN apt-get install -y python3-dev g++
RUN apt-get install -y binutils libgdal-dev gdal-bin 
RUN apt-get install -y python3.10-venv
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal
ENV PYTHONPATH=/app

# Copy project files
COPY pyproject.toml poetry.lock /app/
COPY ./src /app/src

# --- Install Poetry using Python 3.12 ---
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ENV PATH="${PATH}:${POETRY_VENV}/bin"

# --- Install project dependencies ---
RUN poetry install --no-interaction