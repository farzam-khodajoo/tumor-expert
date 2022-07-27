<img src="https://github.com/arshamkhodajoo/tumor-export-frontend/raw/main/public/review.png" />

## About Tumor Expert.
Tumor Expert is a tumor segmentation and classification software built to simplify medical imaging tasks for radiologists.

Gliomas are the most common primary brain tumors, and Magnetic Resonance Imaging (MRI) of brain tumors is critical for progression evaluation, treatment planning, and assessment of this disease.

Tumor Expert takes four sequences of MRI Nifti files, namely T1-Weight, T1CE-Weight, T2-weight, and Fluid Attenuation Inversion Recovery (FLAIR) images, to segment tumors as Whole Tumor, Enhancing Tumor, non-Enhancing tumor.

## Installation and Setup

### Docker Installation
build The Container:

`docker build -t brain-tumor-export <path_to>/tumor-expert`

Run The Container

`docker run -dp 8000:8000 brain-tumor-expert`

### Manual installation
Tumor Expert uses React for Web applications which placed in another repository, thus we need to clone and get the production build first:
```
git clone https://github.com/arshamkhodajoo/tumor-expert-frontend.git
cd tumor-expert-frontend
```

now install dependencies and build app
```
npm install
npm run build
```

At this point, you will have `build/` inside the app folder,
and we will set the ENV variable to the `build/` directory path. We will use it to serve HTML and static files from the fast-API end.
#### Note: run this command only when your current directory is `<path>/tumor-expert-frontend`

for Linux: 

`export REACT_BUILD="$(pwd)/build"`

for PowerShell (Windows): 


```$ENV:REACT_BUILD=pwd
$ENV:REACT_BUILD += "\build"
```

At this point, we are ready to setup Server.
get out of `/tumor-expert-frontend` and run:

```
git clone https://github.com/arshamkhodajoo/tumor-expert.git
cd tumor-expert
```

install dependencies:

`pip install -r requirements.txt`


set trained model path to onnx file place in repositiry:

for Linux:

`export WEIGHTS="$(pwd)/public/brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.onnx"`

for PowerShell (Windows)

```
$ENV:WEIGHTS=pwd
$ENV:WEIGHTS += "/public/brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.onnx"
```

<italic>windows truly sucks, I feel it with each of my nerve cells.</italic>

Run Server:

```
cd bras/app/
uvicorn app:app --port 8000
```

now server is running on `localhost:8000`

open your browser and type `localhost:8000`to load application.
