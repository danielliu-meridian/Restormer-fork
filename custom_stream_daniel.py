import os
import sys
import time

from senxorplus.stark import STARKFilter
from senxor.utils import data_to_frame, remap, connect_senxor, cv_render
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
mi48.regwrite(0xB4, 0x2A)
# see if filtering is available in MI48 and set it up
mi48.regwrite(0xD0, 0x00)   # disable temporal filter
if ncols == 80:
    mi48.regwrite(0x20, 0x00)   # disable STARK filter
mi48.regwrite(0x30, 0x00)   # disable median filter
mi48.regwrite(0x25, 0x00)   # disable MMS

mi48.set_sens_factor(1.0)
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

# cv2.namedWindow("Display")
nrows += 2  # account for the padding 

with torch.no_grad():
    while True:
        data, header = mi48.read()
        if data is None:
            mi48.stop()
            sys.exit(1)

        # img_multiple_of = 8
        # height,width = input_.shape[2], input_.shape[3]
        # H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        # padh = H-height if height%img_multiple_of!=0 else 0
        # padw = W-width if width%img_multiple_of!=0 else 0
        # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        # print(f"resized_image.shape: {input_.shape}")

        padded = np.hstack([data[:80], data, data[-80:]])
        frame = data_to_frame(padded, (ncols, nrows), hflip=hflip)
        min_temp = frame.min()
        max_temp = frame.max()

        # convert the frame to a tensor
        frame_01 = remap(frame, new_range=(0,1), to_uint8=False).astype(float)
        tensor = np.expand_dims(frame_01, axis=2); print('tensor before permutation', tensor.shape)
        tensor = torch.from_numpy(tensor).float().permute(2,0,1).unsqueeze(0)
        print('input tensor shape', tensor.shape)
        print(tensor.dtype)


        # run the model and get the output
        output = model(tensor).squeeze(0).squeeze(0).numpy()
        output = remap(output, curr_range=(0,1), new_range=(min_temp, max_temp), to_uint8=False)
        #output_ = np.dstack((output_, output_, output_))

        img_original = cv_render(remap(frame), resize=(ncols*scale, nrows*scale),
                                 interpolation=cv2.INTER_NEAREST_EXACT,
                                 colormap='rainbow2', with_colorbar=False, display=False)
        img_filtered = cv_render(remap(output), resize=(ncols*scale, nrows*scale),
                                 interpolation=cv2.INTER_NEAREST_EXACT,
                                 colormap='rainbow2', with_colorbar=False, display=False)
        cv2.imshow("Display", np.hstack([img_original, img_filtered]))

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