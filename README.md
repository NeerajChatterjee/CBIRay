# CBIRay

A specialized medical image retrieval website leveraging Content-Based Image Retrieval (CBIR) to locate similar x-ray images for efficient diagnosis and analysis by healthcare professionals.

## Prerequisites
1. [Node.js](https://nodejs.org/)
2. [Python](https://www.python.org/)

## Installation

1. Open a terminal and run `git clone https://github.com/NeerajChatterjee/CBIRay`  in a suitable directory
2. Open a terminal inside the `frontend` folder and run `npm install`
3. Open the `backend` folder and run `pip install -r requirements.txt`
4. In `backend`, make sure the csv files are present in their respective folders
5. Also, make sure that the vgg weights are present inside the `cnn` folder
6. In `backend`, create a `static` folder. Place the chest x-ray files in this folder.
7. Create an `uploads` folder inside the `static` folder.
8. Create a `.env` file inside the `backend` folder. The contents should be like so: `PROJECT_PATH={enter path here}`. Example: `PROJECT_PATH=C:/CBIRay`
9. Inside `backend`, run `main.py`
10. Inside `frontend`, run `npm start`
11. Navigate to `localhost:3000` 