**Deep Image Search** is an AI-based image search engine that includes **CCBR(CNN Classifier Based Retrieval technique)** with **vectorized search technique** or **deep transfer learning features Extraction** with **vectorized search technique**


![Generic badge](https://img.shields.io/badge/AI-Advance-green.svg) ![Generic badge](https://img.shields.io/badge/Python-3.6|3.7-blue.svg) ![Generic badge](https://img.shields.io/badge/pip-v3-red.svg) ![Generic badge](https://img.shields.io/badge/Pytorch-v1-orange.svg) ![Generic badge](https://img.shields.io/badge/TensorFlow-v2-orange.svg) ![Generic badge](https://img.shields.io/badge/scikitlearn-latest-green.svg) ![Generic badge](https://img.shields.io/badge/selenium-latest-green.svg) ![Generic badge](https://img.shields.io/badge/beautifulsoup4-latest-green.svg) ![Generic badge](https://img.shields.io/badge/fastapi-latest-green.svg) ![Generic badge](https://img.shields.io/badge/streamlite-latest-green.svg) ![Generic badge](https://img.shields.io/badge/dvc-latest-green.svg)



<h2><img src="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence9-512.png" alt="Brain+Machine" height="38" width="38"> Creators </h2>

### [Mohan Kumar](https://github.com/kkkumar2?tab=repositories)

### Sandeep Jena

# About the Project:
    
This project can be applied to recommend images based on the image you upload. The technique used here are
    
    1) CCBR - This technique is a custom trained model which classifies the incoming image and extracts the features    parallely and recommends top 5 images by using KNN. This technique is handled in both Tensorflow and Pytorch frameworks

    2) Normal feature Extraction using pretrained model - This technique uses VGG16 in Tensorflow and RESNET50 model in Pytorch framework. This Method will not classify image into category and use that category for selecting top 5 images instead we search the Entire images in all category using ANNOY Library(C++ Library with python bindings)
    
    
#### If you don't want download (C++ Libary) used sklearn libary  insted of ANNOY library There is another [repro](https://github.com/sandeepjena7/Image-Based-Recommendation-System)
        
## **KNN - K nearest Neighbours:**
**input** - features vectors for the class which the model predicted while fitting and uploaded image's feature vector while predicting

**output** - top 5 images Index and distance

```python
featurevector = self.FE.extract(img)
neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbors.fit(self.vectors)
distances, indices = neighbors.kneighbors([featurevector])
```
    
## **ANNOY - Approximate nearest Neighbours oh yeah:**
**input** - Create a index with the help of all images and use that index while predicting

**output** - top 5 images Index 

Build Index:
```Python
f = features[0].shape[0]
t = AnnoyIndex(f, 'angular')
for i in range(len(features)):
    v = features[i]
    t.add_item(i, v)

t.build(30) 
t.save(self.annoy_path)
```
Predict:
```Python
featurevector = self.FE.extract(img)
indices = self.u.get_nns_by_vector(featurevector, 5, search_k=30)

```


How to use this Project in your local:

Clone the Repository using
```bash
git clone https://github.com/kkkumar2/Image_Recommendation_System.git
```
Create a environment and install all the requirements from requirements.txt
```bash
conda env create -f environment.yml
```
**OR**

```bash
bash initial_setup.sh
```
Activate enviroment
```bash
source activate recomendation
```

Run the following command to run the web application locally
```bash
streamlit run Streamlit.py
```
A demo is  shown below:

![GIF](recommendation.gif)

## Dataset

<a href="https://drive.google.com/drive/folders/1iReMDMw_WSyuLTXXWQv7H0jMv2e4Wsqd?usp=sharing">Click here</a> to get the dataset of resized 5000 images from myntra
