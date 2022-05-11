import argparse
import os
from image_resize import letterbox
import cv2
from tqdm import tqdm

class Resize(object):
    def __init__(self,image_path,output_path):
        self.image_path = image_path
        self.output_path = output_path

    def convert(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        folders = ['train','val']
        for ele in folders:
            for r, d, f in tqdm(os.walk(os.path.join(self.image_path,ele)),colour="GREEN",desc=f"Working with {ele} folder"):
                for dir in d:
                        dir_path = os.path.join(self.image_path,ele,dir)
                        for img in tqdm(os.listdir(dir_path),colour="MAGENTA",desc=f"Resizing for {dir}"):
                            dst_dir = os.path.join(self.output_path,ele,dir)
                            if not os.path.exists(dst_dir):
                                os.makedirs(dst_dir)
                            inp_path = os.path.join(dir_path,img)
                            out_path = os.path.join(dst_dir,img)
                            im = cv2.imread(inp_path)
                            out_img = letterbox(im)
                            cv2.imwrite(out_path,out_img)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',  type=str, default=os.path.join('Artifacts','data','raw'), help='Image path')
    parser.add_argument('--output_path', type=str, default=os.path.join('Artifacts','data','pre_processed'), help='Ouput path')

    opt = parser.parse_args()
    return opt

def main(opt):
    obj1 = Resize(**vars(opt))
    obj1.convert()
    print("Resizing done sucessfully")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
