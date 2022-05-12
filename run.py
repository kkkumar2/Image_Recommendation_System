from DeepImageSearch.Normalfeatureextractor import SearchImageNormalPytoch,SearchImageNormaltensorflow
from DeepImageSearch.Ccbrfeatureextractor import SearchImagePytorch,SearchImageTensorflow
from DeepImageSearch.utils.allutils import util
import pandas as pd
import os
from tqdm import tqdm
from os.path import exists

class RUN:
    def __init__(self,config,params):
        # self.pyccbr = SearchImagePytorch(config,params)
        # self.tfcbbr = SearchImageTensorflow(config)
        # self.pynormal = SearchImageNormalPytoch(config,params)
        self.tfnormal = SearchImageNormaltensorflow(config)
        self.config = config
        self.params = params
        config = util.read_yaml(config)
        artifact_dir = config["artifacts"]["artifactdir"]
        metadata_dir = config["artifacts"]["meta_data_dir"]
        analysisdataframe = config['artifacts']['analysisdataframe']
        data_path = os.path.join(artifact_dir,metadata_dir,analysisdataframe)
        dict_ = util.read_pickle(data_path)
        self.df = pd.DataFrame(dict_)
        image_dir = config["artifacts"]["image_dir"]
        preprocessed = config["artifacts"]["preprocessed"]
        raw = config["artifacts"]["raw"]
        train = config["artifacts"]["train_dir"]
        val = config["artifacts"]["val_dir"]
        self.basepath1 = os.path.join(artifact_dir,image_dir,raw,train)
        self.basepath2 = os.path.join(artifact_dir,image_dir,raw,val)



    def search(self,Framework,Technique,image):

        if Framework == "Pytroch":
            if Technique == "ccbr": 
                imagespath = SearchImagePytorch(self.config,self.params).get_similar_images(image)
            else:
                imagespath = SearchImageNormalPytoch(self.config,self.params).get_similar_images_normal_Annoy(image)

        else:
            if Technique == 'ccbr':
                imagespath = SearchImageTensorflow(self.config).get_similar_images(image)
            else:
                imagespath = SearchImageNormaltensorflow(self.config).get_similar_images_normal_Annoy(image)
                            
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
    # https://stackoverflow.com/questions/56658723/how-to-maintain-order-when-selecting-rows-in-pandas-dataframe  
            ele = self.df.set_index('Imagename').reindex(base_imgname).reset_index()
            img_names = imagespath

        elif type == 'rating_based':
            base_name = imagespath[0].split('\\')[-1]
            Category_name = self.df[self.df['Imagename'] == base_name]['Category'].values
            sorted_df = self.df.sort_values("Price", ascending=True)
            sorted_df = sorted_df.sort_values("Ratting", ascending=False)
            sorted_df = sorted_df.groupby('Category').get_group(Category_name[0]).head(5)
            img_names = sorted_df['Imagename'].to_list()
            img_names = self.full_imagepath_fetch(Category_name[0],img_names)
            ele = sorted_df

        elif type == 'count_based':
            base_name = imagespath[0].split('\\')[-1]
            Category_name = self.df[self.df['Imagename'] == base_name]['Category'].values
            sorted_df = self.df.sort_values("NoOfPurchasing", ascending=False)
            sorted_df = sorted_df.groupby('Category').get_group(Category_name[0]).head(5)
            img_names = sorted_df['Imagename'].to_list()
            img_names = self.full_imagepath_fetch(Category_name[0],img_names)
            ele = sorted_df

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
            image_path = os.path.join(os.getcwd(),self.basepath1,Category_name,ele)
            image_path2 = os.path.join(os.getcwd(),self.basepath2,Category_name,ele)
            if exists(image_path):
                img_names.append(image_path)
            elif exists(image_path2):
                img_names.append(image_path2)
            else:
                continue
        return img_names


if __name__ == '__main__':
    j = RUN('config/config.yaml','params.yaml')
    