# syntax=docker/dockerfile:1

FROM python:3.8-slim

# install git command to clone frontend source
RUN apt-get update
RUN apt-get install -y --no-install-recommends git
# install npm command
RUN apt-get install -y --no-install-recommends nodejs npm

WORKDIR /app
# move dependency file
COPY requirements.txt requirements.txt
# install all dependencies
RUN pip install -r requirements.txt
# copy source code files into app/
COPY . app/
# download react front end from repository
RUN git clone https://github.com/arshamkhodajoo/tumor-export-frontend.git frontend
# install react dependencies and get production build
WORKDIR /app/frontend
RUN npm install
RUN npm run build
# set enviroment variables
# path to openVino onnx model weights
ARG WEIGHTS=app/public/brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.onnx
# path to react app production build
ARG READ_BUILD=app/frontend/build

# run ASGI server on localhost:8000
WORKDIR /app/bras/app
RUN uvicorn app:app --port 8000