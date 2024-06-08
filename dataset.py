import sys
sys.path.append('/home/lab5017/projects/SAM_CLIP/CLIP_emergencies/v1/')
from CLIP import load_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torch.utils.data import Dataset
from utils_stream import str_to_time
from sklearn.metrics import accuracy_score, precision_score


class VideoDataset(Dataset):
    def __init__(self, excel_path='', test=False, test_split=None, video_path='',
                 part_of_data=None, verbose=False, video_per_class=None):
        self.verbose = verbose
        self.excel_path = excel_path
        self.test_split = test_split
        self.test_mode = test
        if self.test_mode:
            self.annotation_name = 'test.xlsx'
            self.emb_path = 'test'
        else:
            self.annotation_name = 'train.xlsx'
            self.emb_path = 'train'
        self.classes = ['Arson', 'Assault', 'Explosion', 'Fighting', 'RoadAccidents', 'Vandalism', 'Normal']
        self.video_path = video_path
        self.preprocess = Compose([Resize((224, 224), interpolation=Image.BICUBIC), ToTensor()])
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        status = self.generate_annotation(excel_path)

        if part_of_data is not None and status:
            self.split_data(part_of_data)
        if status:
            self.annotation.to_excel(os.path.join(excel_path, self.annotation_name))
            self.image_encodel = load_model('/home/lab5017/projects/SAM_CLIP/clip/checkpoint/model.pt').cuda()
            self.generate_embeddings()
        if test:
            self.results = pd.DataFrame()
        if video_per_class is not None:
            self.extract_videos_by_class(video_per_class)

    def generate_annotation(self, path_to_results):
        if os.path.isfile(os.path.join(path_to_results, self.annotation_name)):
            self.annotation = pd.read_excel(os.path.join(path_to_results, self.annotation_name))
            return False
        else:
            self.get_list_excel_files(path_to_results)
            self.annotation = pd.DataFrame()
            for excel_path in self.list_file_excel:
                excel = self.parse_excel(excel_path)
                self.annotation = pd.concat([self.annotation, excel], axis=0)
                if self.verbose:
                    print(self.annotation)
            return True

    def generate_embeddings(self):
        for idx in range(self.__len__()):
            frame, label = self.get_frame(idx)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = self.preprocess(image)
            image -= self.image_mean[:, None, None]
            image /= self.image_std[:, None, None]
            embedding = self.img_to_embedding(torch.unsqueeze(image, 0))[0]
            os.makedirs(f'{self.excel_path}/{self.emb_path}', exist_ok=True)
            torch.save(embedding, f'{self.excel_path}/{self.emb_path}/{idx}.pth')
    
    def __getitem__(self, idx):
        embedding = torch.load(f'{self.excel_path}/{self.emb_path}/{idx}.pth', map_location='cpu')
        label = self.annotation.iloc[idx]['gt']
        return embedding, label, idx

    def get_list_excel_files(self, excel_path):
        self.list_file_excel = []
        for name_class in self.classes:
            path_to_excel = os.path.join(excel_path, name_class)
            if os.path.exists(path_to_excel):
                list_files = os.listdir(path_to_excel)
                list_files.sort()
                if self.test_split is not None:
                    if self.test_mode:

                        self.list_file_excel.extend([os.path.join(path_to_excel,
                                                                  i) for i in list_files[-self.test_split:]])
                    else:

                        self.list_file_excel.extend([os.path.join(path_to_excel,
                                                                  i) for i in list_files[:-self.test_split]])
                else:

                    self.list_file_excel.extend([os.path.join(path_to_excel, i) for i in list_files])
        if self.verbose:
            print(self.list_file_excel)

    def parse_excel(self, path: str, gt_only: bool = True):
        video_name = path.rsplit('/', 1)[1].split('x264')[0] + 'x264.mp4'
        class_name = path.rsplit('/', 2)[1]
        res_df = pd.read_excel(path)
        if res_df.columns[0] == 'Unnamed: 0':
            res_df.columns = [None] + list(res_df.columns[1:])
            res_df.set_index(res_df.columns[0], inplace=True)
        if gt_only:
            cpgt_df = pd.DataFrame()
            col = 'gt'
            if col in res_df.columns:
                cpgt_df = pd.concat([cpgt_df, res_df[col]], axis=1)
                cpgt_df['time'] = cpgt_df.index.map(str_to_time)
                cpgt_df[col] = cpgt_df[col].map(lambda cls: self.classes.index(class_name) if cls == 1 else
                                                self.classes.index('Normal'))
                cpgt_df['video_name'] = [class_name + '/' + video_name for x in range(len(cpgt_df))]
            return cpgt_df
        else:
            return res_df
    
    def img_to_embedding(self, image_tensor):
        with torch.no_grad():
            image_features = self.image_encodel.encode_image(image_tensor).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def extract_videos_by_class(self, number_videos):
        small_annotation = pd.DataFrame()
        list_videos = list(pd.unique(self.annotation['video_name']))
        for idx_class in range(len(self.classes) - 1):
            part_video = self.annotation.loc[self.annotation['gt'] == idx_class]
            list_videos = list(pd.unique(part_video['video_name']))
            list_videos = random.sample(list_videos, number_videos)
            for name_video in list_videos:
                small_annotation = pd.concat([small_annotation, 
                                              self.annotation.loc[self.annotation['video_name'] == name_video]],
                                             axis=0)

        self.annotation = small_annotation

    def split_data(self, part_of_data):
        small_annotation = pd.DataFrame()
        list_videos = list(pd.unique(self.annotation['video_name']))
        for video in list_videos:
            part_video = self.annotation.loc[self.annotation['video_name'] == video]
            part_anomaly = part_video.loc[part_video['gt'] != self.classes.index('Normal')]
            part_normal = part_video.loc[part_video['gt'] == self.classes.index('Normal')]
            number_sample_anomaly = min(round(part_of_data * len(part_anomaly)), len(part_normal))
            number_sample_normal = min(len(part_normal), round(number_sample_anomaly / len(self.classes)))
            small_annotation = pd.concat([small_annotation, part_anomaly.sample(number_sample_anomaly)], axis=0)
            small_annotation = pd.concat([small_annotation, part_normal.sample(number_sample_normal)], axis=0)
        if self.verbose:
            print(len(small_annotation))
        self.annotation = small_annotation
    
    def get_targets(self, ):
        return self.annotation['gt'].to_list()
    
    def predict_to_frame(self, indexes:list, predict:list):
        ann = self.annotation.iloc[indexes]
        for name_class in self.classes:
            ann[name_class] = [i[self.classes.index(name_class)] for i in predict]
        self.results = pd.concat([self.results, ann], axis=0)
    
    def save_predict(self, path):
        y_true = self.results['gt'].to_numpy()
        y_pred = np.argmax(self.results[self.classes].to_numpy(), axis=1)
        print(precision_score(y_true, y_pred, average=None))
        self.results.to_excel(os.path.join(path))
        
    def get_frame(self, idx):
        ann = self.annotation.iloc[idx]
        video_capture = None
        for i in range(1, 5):
            full_path = os.path.join(self.video_path, f'anomaly_part-{i}/videos/', ann['video_name'])
            if os.path.isfile(full_path):
                if self.verbose:
                    print(full_path)
                video_capture = cv2.VideoCapture(full_path)
        assert video_capture is not None

        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate = round(video_capture.get(cv2.CAP_PROP_FPS))
        frame_number = int(ann['time'] * frame_rate)
        if frame_number > total_frames - 10:
            frame_number = total_frames - 10
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video_capture.read()
        return frame, ann['gt']

    def get_class_distributed(self, ):
        return dict(sorted(self.annotation['gt'].value_counts().to_dict().items()))
    
    def __len__(self, ):
        return len(self.annotation['gt'])


if __name__ == '__main__':
    data = VideoDataset(excel_path='/home/lab5017/projects/SAM_CLIP/CLIP_emergencies/v1/results',
                        video_path='/home/lab5017/projects/SAM_CLIP/tracking_anything/assets',
                        verbose=True)
    print(data[100])
    print(data[1000])

