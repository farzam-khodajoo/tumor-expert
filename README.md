<img src="https://github.com/arshamkhodajoo/tumor-export-frontend/raw/main/public/review.png" />

## About Tumor Expert.
Tumor Expert is a tumor segmentation and classification software built to simplify medical imaging tasks for radiologists.

Gliomas are the most common primary brain tumors, and Magnetic Resonance Imaging (MRI) of brain tumors is critical for progression evaluation, treatment planning, and assessment of this disease.

Tumor Expert takes four sequences of MRI Nifti files, namely T1-Weight, T1CE-Weight, T2-weight, and Fluid Attenuation Inversion Recovery (FLAIR) images, to segment tumors as Whole Tumor, Enhancing Tumor, non-Enhancing tumor.

## Toturial
### Upload Files
<img src="https://github.com/arshamkhodajoo/tumor-expert-frontend/blob/main/public/upload-guid.gif" width="300px"/>

application expects four MRI sequences (T1, T2, T1CE, FLAIR). to classify files, corresponding keyword should be placed in each filename

for example:

`case_something_t1.nii.gz` will be interpreted as T1 weight

and `something_flair.nii.gz` as FLAIR.

### View, Scale, Segment
<img src="https://github.com/arshamkhodajoo/tumor-expert-frontend/blob/main/public/view-expert.gif" width="300px">
view options are on the top left corner, you can switch view from sequence weights, hide segmentation and scale image on screen

and slicer on the bottom.

## Installation and Setup

### Docker Installation
build The Container:

`docker build -t brain-tumor-expert <path_to>/tumor-expert`

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


```
$ENV:REACT_BUILD=pwd
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

install source as local package:
`pip install -e .`

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

## Deep Learning Model
An 3D U-net structure were used to segment tumor spot on input images.

For production, Tumor Expert uses [OpenVino's open 3D brain tumor segmentation model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/brain-tumor-segmentation-0002) and inference model is optimized for Intel CPUs

**There is** a custome training pipline if you like to train locally, please check `notebooks/colab_brats_train.ipynb`

Please check `config/unet.yaml` for model structure and train pipeline settings,
model hyperparameters were taken from this [paper](https://arxiv.org/abs/2110.03352)


## API 
API Interface also can be used separately, you can use inference call at `localhost:8000/view` to get segmentation nifti file.

example:
```javascript
const data = {
  t1: File(...),
  t1ce: File(...),
  t2: File(...),
  flair: File(...)
}

const form = new Form()
for (const [key, value] of Object.entries(data)) {
        form.append(key, value)
}

axios({
        method: "POST",
        url: "localhost:8000/views/",
        data: form,
        headers: {
            "Content-Type": "multipart/form-data"
        },
        responseType: 'blob'
    })
```
