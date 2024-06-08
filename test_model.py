import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dataset import VideoDataset
from model import LogisticRegressionTorch
from tqdm import tqdm

model = LogisticRegressionTorch(input_size=512, num_classes=7).to('cuda')

test_data = VideoDataset(excel_path='/home/lab5017/projects/SAM_CLIP/CLIP_emergencies/v1/results',
                         video_path='/home/lab5017/projects/SAM_CLIP/tracking_anything/assets',
                         test=True)
train_dataloader = DataLoader(test_data, batch_size=1000, num_workers=10)

model.load('ckpt/mlp/28_05_50.46_linear_model29_mlp.pth')
model.eval()
epoch_number = 0

for data in tqdm(train_dataloader):
    inputs, labels, idx = data
    inputs = torch.squeeze(inputs, 1)
    outputs = model(inputs.cuda()).cpu()
    test_data.predict_to_frame(idx.tolist(), outputs.tolist())
    _, predicted = torch.max(outputs.data.cpu(), 1)
    correct = (predicted == labels).sum()
    accuracy = 100 * (correct.item()) / len(inputs)
    print(accuracy)

test_data.save_predict('results_mlp.xlsx')
