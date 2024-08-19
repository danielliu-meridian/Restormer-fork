import os
import sys
import time

from senxorplus.stark import STARKFilter
from senxor.utils import data_to_frame, remap, connect_senxor
from senxor.filters import RollingAverageFilter
# from senxor.display import cv_render

import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
from runpy import run_path
import cv2
# from tqdm import tqdm
# import argparse
import numpy as np

import torch.nn.functional as F

def get_weights_and_parameters(task, parameters):
    if task == 'denoise':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
        return weights, parameters
    else:
        return None, None


# Get model weights and parameters
parameters = {'inp_channels':1, 'out_channels':1, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
weights, parameters = get_weights_and_parameters("denoise", parameters)

load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

checkpoint = torch.load(weights, map_location=torch.device('cpu'))
params = checkpoint['params']
model.load_state_dict(params)
model.eval()

mi48 = connect_senxor()
ncols, nrows = mi48.fpa_shape

# set desired FPS
mi48.regwrite(0xB4, 0x02)
# see if filtering is available in MI48 and set it up
mi48.regwrite(0xD0, 0x00)   # disable temporal filter
if ncols == 80:
    mi48.regwrite(0x20, 0x00)   # disable STARK filter
mi48.regwrite(0x30, 0x00)   # disable median filter
mi48.regwrite(0x25, 0x00)   # disable MMS

mi48.set_sens_factor(100)
mi48.set_offset_corr(0.0)

# initiate continuous frame acquisition
time.sleep(1)
with_header = True
mi48.start(stream=True, with_header=with_header)

minav = RollingAverageFilter(N=25)
maxav = RollingAverageFilter(N=16)
minav2 = RollingAverageFilter(N=25)
maxav2 = RollingAverageFilter(N=16)

stark_par = {'sigmoid': 'sigmoid',
# 'variant': 'original',
'lm_atype': 'ra',
'lm_ks': (5,5),
'lm_ad': 9,
'alpha': 2.0,
'beta': 2.0,}
frame_filter = STARKFilter(stark_par)

scale = 2
hflip = False

mask = np.ones((nrows, ncols), dtype=bool)

# cv2.namedWindow("Display")

with torch.no_grad():
    while True:
        data, header = mi48.read()
        if data is None:
            mi48.stop()
            sys.exit(1)

        frame = data_to_frame(data, (ncols, nrows), hflip=hflip)
        # corner_temp = frame[mask].mean()
        # frame[mask==False] = corner_temp

        # rolling average filter
        min_temp = minav(np.median(np.sort(frame.flatten())[:16]))
        max_temp = maxav(np.median(np.sort(frame.flatten())[-5:]))
        frame = np.clip(frame, min_temp, max_temp)
        # STARK filter
        frame = frame_filter(frame)
        # rolling average filter
        min_temp = minav(np.median(np.sort(frame.flatten())[:9]))
        max_temp = maxav(np.median(np.sort(frame.flatten())[-5:]))
        frame = np.clip(frame, min_temp, max_temp)
        # print(f"frame.shape: {frame.shape}")

        # convert the frame to a tensor
        input_ = np.expand_dims(remap(frame), axis=2)
        input_ = torch.from_numpy(input_).float().div(255.).permute(2,0,1).unsqueeze(0)
        print(f"input_.shape: {input_.shape}")

        # resize the input image to a size that is divisible by 2
        # Pad the input if not_multiple_of 8
        img_multiple_of = 8
        height,width = input_.shape[2], input_.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        print(f"resized_image.shape: {input_.shape}")

        # run the model and get the output
        output_ = remap(model(input_).squeeze(0).squeeze(0).numpy())
        output_ = np.dstack((output_, output_, output_))

        # resize the output image
        scale = 5
        output_ = cv2.resize(output_, (ncols*scale, nrows*scale), interpolation=cv2.INTER_NEAREST_EXACT)
        frame = cv2.resize(remap(frame), (ncols*scale, nrows*scale), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow("Display", output_)
        cv2.imshow("Original", frame)

        key = cv2.waitKey(1)  # & 0xFF
        if key == ord("q"):
            break
        if key == ord("o"):
            use_frame_offset = not use_frame_offset
        if key == ord("0"):
            show_frame_offset = not show_frame_offset
        if key == ord("f"):
            hflip = not hflip

# stop capture and quit
mi48.stop()
cv2.destroyAllWindows()
