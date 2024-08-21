import os
import sys

import torch
from runpy import run_path
import numpy as np
import cv2 as cv
from senxor.utils import remap

##### MODEL SET UP
def get_weights_and_parameters(task, parameters):
    if task == 'denoise':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
        return weights, parameters
    else:
        return None, None


# Get model weights and parameters
parameters = {'inp_channels':1, 'out_channels':1, 'dim':48, 'num_blocks':[4,6,6,8],
              'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66,
              'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}

weights, parameters = get_weights_and_parameters("denoise", parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

checkpoint = torch.load(weights, map_location=torch.device('cpu'))
params = checkpoint['params']
model.load_state_dict(params)
model.eval()
#####

ncols, nrows = 160, 120
scale = 2

videoroot = "C:/Users/takao/Desktop/denoising_raw_data"
videonames = os.listdir(videoroot)
videopath = os.path.join(videoroot, videonames[0])
cap = cv.VideoCapture(videopath)

fourcc = cv.VideoWriter_fourcc(*'MJPG')
print(f"processing {videonames[0]}")
outpath = os.path.join(videoroot, videonames[0][:-4]+"_processed.avi")
out = cv.VideoWriter(outpath, fourcc, 20.0, (ncols * scale,  nrows * scale))

counter = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        print(f"gray.shape: {gray.shape}")
        input_ = np.expand_dims(gray, axis=2)
        print(f"input_.shape: {input_.shape}")
        input_ = torch.from_numpy(input_).float().div(255.).permute(2,0,1).unsqueeze(0)
        output_ = remap(model(input_).squeeze(0).squeeze(0).numpy())

        output_ = np.dstack((output_, output_, output_))
        out.write(output_)
        print(f"saved frame idx {counter}")
        counter += 1

cap.release()
out.release()
cv.destroyAllWindows()
