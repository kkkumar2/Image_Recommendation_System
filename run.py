from DeepImageSearch.Normalfeatureextractor import SearchImageNormalPytoch,SearchImageNormaltensorflow
from DeepImageSearch.Ccbrfeatureextractor import SearchImagePytorch,SearchImageTensorflow
from DeepImageSearch.utils.allutils import util
import pandas as pd
import os
from tqdm import tqdm
from os.path import exists

class RUN:
    def __init__(self,config,params):
        self.pyccbr = SearchImagePytorch(config,params)
        self.tfcbbr = SearchImageTensorflow(config)
        self.pynormal = SearchImageNormalPytoch(config,params)
        self.tfnormal = SearchImageNormaltensorflow(config)
        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        metadata_dir = config["artifacts"]["meta_data_dir"]
        analysisdataframe = config['artifacts']['analysisdataframe']
        data_path = os.path.join(artifact_dir,metadata_dir,analysisdataframe)
        dict_ = util.load_pickle(data_path)
        self.df = pd.DataFrame(dict_)
        image_dir = config["artifacts"]["image_dir"]
        preprocessed = config["artifacts"]["preprocessed"]
        raw = config["artifacts"]["raw"]
        train = config["artifacts"]["train"]
        val = config["artifacts"]["val"]
        self.basepath1 = os.path.join(artifact_dir,image_dir,preprocessed,raw,train)
        self.basepath2 = os.path.join(artifact_dir,image_dir,preprocessed,raw,val)



    def search(self,Framework,Technique,image):

        if Framework == "Pytroch":
            if Technique == "ccbr": 
                imagespath = self.pyccbr.get_similar_images(image)
            else:
                imagespath = self.pynormal.get_similar_images_normal(image)

        else:
            if Technique == 'ccbr':
                imagespath = self.tfccbr.get_similar_images(image)
            else:
                imagespath = self.tfnormal.get_similar_images_normal(image)

        out1 = self.image_based(imagespath,'image_based')
        out2 = self.image_based(imagespath,'rating_based')
        out3 = self.image_based(imagespath,'count_based')

        return out1,out2,out3

    def image_based(self,imagespath,type):

        base_imgname = []; URL = []
        baselink = 'https://www.myntra.com/'
        if type == 'image_based':
            for img_name in imagespath:
                base_imgname.append(img_name.split('\\')[-1])        
            ele = self.df[self.df['Imagename'].isin(base_imgname)]
            img_names = imagespath

        elif type == 'rating_based':
            base_name = imagespath[0].split('\\')[-1]
            Category_name = self.df[self.df['Imagename'] == base_name]['category']
            sorted_df = self.df.sort_values("Price", ascending=True)
            sorted_df = sorted_df.sort_values("Ratting", ascending=False)
            sorted_df = sorted_df.groupby('Category').get_group(Category_name).head(5)
            img_names = sorted_df['Imagename'].to_list()
            img_names = self.full_imagepath_fetch(Category_name,img_names)

        elif type == 'count_based':
            base_name = imagespath[0].split('\\')[-1]
            Category_name = self.df[self.df['Imagename'] == base_name]['category']
            sorted_df = self.df.sort_values("NoOfPurchasing", ascending=False)
            sorted_df = sorted_df.groupby('Category').get_group(Category_name).head(5)
            img_names = sorted_df['Imagename'].to_list()
            img_names = self.full_imagepath_fetch(Category_name,img_names)

        b_URL = ele['WebsiteProductLink'].to_list()
        URL = [baselink+i for i in b_URL]
        Price = ele['Price'].to_list()
        rating = ele['Ratting'].to_list()
        Buy_count = ele['NoOfPurchasing'].to_list()
        BrandName = ele['BrandName'].to_list()
        
        return {type:[img_names,URL,Price,rating,Buy_count,BrandName]}
            
    def full_imagepath_fetch(self,Category_name,img_base):
        img_names = []
        for ele in img_base:
            image_path = os.path.join(self.basepath1,Category_name,ele)
            image_path2 = os.path.join(self.basepath2,Category_name,ele)
            if exists(image_path):
                img_names.append(image_path)
            elif exists(image_path2):
                img_names.append(image_path2)
            else:
                continue
        return img_names


if __name__ == '__main__':
    j = RUN('config/config.yaml','params.yaml')
    