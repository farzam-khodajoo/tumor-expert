# syntax=docker/dockerfile:1

FROM python:3.8-slim

# install git command to clone frontend source
RUN apt-get update
RUN apt-get install -y --no-install-recommends git curl
# install npm command and node ^16
RUN curl -s https://deb.nodesource.com/setup_16.x | bash
RUN apt-get install nodejs -y

WORKDIR /app
# move dependency file
COPY requirements.txt requirements.txt
# install all dependencies
RUN pip install -r requirements.txt
# just a trick, let it be ..
RUN pip install python-multipart
# copy source code files into app/
COPY . .
# download react front end from repository
RUN git clone https://github.com/arshamkhodajoo/tumor-export-frontend.git frontend
# react codes cloned  into /app/frontend
WORKDIR /app/frontend
# install react dependencies
RUN npm install
# get production build from react app
RUN npm run build


# install source code as python package
WORKDIR /app
RUN pip install -e .

# set path to react production build which includes index.html
ENV READ_BUILD=frontend/build
# set path to onnx model
ENV WEIGHTS=public/brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.onnx
# temporary directory, only to store temporary files
ENV TEMP=/tmp/foo

# run ASGI server on 0.0.0.0:8000
CMD ["uvicorn", "bras.app.app:app", "--host", "0.0.0.0", "--port", "8000"]