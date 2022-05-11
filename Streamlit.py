import streamlit as st
import cv2 
import numpy as np
from PIL import Image
import os
import base64
from DeepImageSearch.utils.allutils import save_uploaded_file
from run import RUN

obj1 = RUN('config/config.yaml','params.yaml')

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" width="114" 
     height="152"/>
        </a>'''
    return html_code

## Function to display images with metadata and add hyperlink to it 
def image_displayer(img_name,URL,Price,rating,Buy_count,BrandName):
    cont2 = st.container()
    i = -1
    for col in cont2.columns(len(img_name)):
        i+=1
        with col:
            gif_html = get_img_with_href(img_name[i], URL[i])
            st.markdown(gif_html, unsafe_allow_html=True)
            st.write(f"{BrandName[i]}")
            st.write(f"{Buy_count[i]} People Bought this")
            st.write(f"Rating :{rating[i]}")
            st.write(Price[i])

def image_parser(out1,out2):
    img_name,URL,Price,rating,Buy_count,BrandName = out1['image_based']         
    st.header("Image based recommendation")
    image_displayer(img_name,URL,Price,rating,Buy_count,BrandName) 

    img_name,URL,Price,rating,Buy_count,BrandName = out2['rating_based'] 
    st.header("Rating based recommendation")
    image_displayer(img_name,URL,Price,rating,Buy_count,BrandName) 

    # img_name,URL,Price,rating,Buy_count,BrandName = out3['count_based'] 
    # st.header("Most People bought this")
    # image_displayer(img_name,URL,Price,rating,Buy_count,BrandName) 

## Gif from local file
def image_converter(filename):
    file_ = open(filename, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url

def find_similar():
    print("Prediction started")
          


st.title("Recomendation")
app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Run on Image on prediction']
)

if app_mode =='About App':

    st.write('In this application we are recommending images based on input image')  

    data_url = image_converter('recommendation.gif')
    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif" width="700" height="400">',unsafe_allow_html=True)


elif app_mode == "Run on Image on prediction":

    app_mode2 = st.sidebar.selectbox('Choose the Framework',
    ["Choose","Tensorflow","Pytorch"]
    )
    if app_mode2 == 'Pytorch' or app_mode2 == "Tensorflow":
        app_mode3 = st.sidebar.selectbox('Choose the Model',
        ["Choose","Pre-trained",'CCBR']
        )
        if app_mode3 == 'Pre-trained' or app_mode3 == 'CCBR':
            uploaded_file = st.sidebar.file_uploader(label="Upload an image", type=[ "jpg", "jpeg",'png'],on_change=find_similar)

            if uploaded_file is not None:

                if save_uploaded_file(uploaded_file):

                    image = np.array(Image.open(uploaded_file))   
                    st.sidebar.text("orginal image")
                    st.sidebar.image(image)
                    
                    out1,out2 = obj1.search(Framework=app_mode2,Technique=app_mode3,image=image)
                    image_parser(out1,out2)

                    st.snow()
                    st.balloons()

                else:
                    print("File is not proper")