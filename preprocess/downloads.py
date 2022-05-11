import pandas as pd
import requests
from tqdm import tqdm
import os, random
import shutil


class Scraping:
        def __init__(self,df):
                self.img_path = os.path.join('Artifacts','data','raw')
                self.category = df['Category']
                self.Imagename = df['Imagename']
                self.ImageLink = df['ImageLink']
                self.src = os.path.join(self.img_path,'train')
                self.dst = os.path.join(self.img_path,'val')
        
        
        def downloads(self,category,image_name,URL):
                
                img_path = os.path.join(self.img_path,'train',category)
                os.makedirs(img_path,exist_ok=True)
                img_path = os.path.join(img_path,image_name)
                response = requests.get(URL)
                file = open(img_path, "wb")
                file.write(response.content)
                file.close()

        
        def download_split_setup(self):
                for i in tqdm(range(len(self.ImageLink)),colour="GREEN",desc=f"Scraping data from Myntra" ):
                        self.downloads(self.category[i],self.Imagename[i],self.ImageLink[i])
                self.train_val_split()
                print("Sucessfuly downloaded data and splitted")

        
        def train_val_split(self):
    
                for r, d, f in os.walk(self.src):
                        for dir in d:
                                dir_path = os.path.join(self.src,dir)
                                tot_len = len(os.listdir(dir_path))
                                fetch_count = int(tot_len * 0.2)
                                for i in range(fetch_count):
                                        random_file = random.choice(os.listdir(dir_path))
#                                        print(random_file)
                                        dst_dir = os.path.join(self.dst,dir)
                                        if not os.path.exists(dst_dir):
                                                os.makedirs(dst_dir)
                                        inp_path = os.path.join(self.src,dir,random_file)
                                        out_path = os.path.join(self.dst,dir,random_file)
                                        shutil.move(inp_path,out_path)



if __name__ == "__main__":
        df = pd.read_pickle('preprocess.pkl')
        print(df['Category'].value_counts())
        obj1 = Scraping(df).download_split_setup()