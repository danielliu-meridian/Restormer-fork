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
import cv2
# from tqdm import tqdm
# import argparse
import numpy as np

import torch.nn.functional as F

from restormer_model import get_weights_and_parameters
from restormer_model import load_model


# set up some variables
scale = 2
task = "denoise" # task name

# Based on the task, get the weights and parameters
weights, parameters = get_weights_and_parameters(task)

# Load the model
model = load_model(parameters, weights)

# connect to the MI48 sensor
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

with torch.no_grad():
    while True:
        data, header = mi48.read()
        if data is None:
            mi48.stop()
            sys.exit(1)
        print(f"header: {header}")
        print(f"data.shape: {data.shape}")

        # img_multiple_of = 8
        # height,width = input_.shape[2], input_.shape[3]
        # H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        # padh = H-height if height%img_multiple_of!=0 else 0
        # padw = W-width if width%img_multiple_of!=0 else 0
        # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        # print(f"resized_image.shape: {input_.shape}")

        # Pad the input if not_multiple_of 8
        # hstack the data to account for the padding
        padded = np.hstack([data[:80], data, data[-80:]])
        # convert the data to a frame
        # account for the padding
        frame = data_to_frame(padded, (ncols, nrows + 2))
        # get the min and max temperature values in the frame
        min_temp = frame.min()
        max_temp = frame.max()

        # convert the frame to a tensor
        frame_01 = remap(frame, new_range=(0,1), to_uint8=False).astype(float)
        tensor = np.expand_dims(frame_01, axis=2); print('tensor before permutation', tensor.shape)
        tensor = torch.from_numpy(tensor).float().permute(2,0,1).unsqueeze(0)
        print('input tensor shape', tensor.shape)
        print(tensor.dtype)


        # run the model and get the output

        # the input type is tensor
        # the input shape is (1, 1, x, x) for grayscale images and (1, 3, x, x) for RGB images
        # the input dtype is float32
        # the output type is tensor
        # the output shape is (1, 1, x, x) for grayscale images and (1, 3, x, x) for RGB images
        # the output dtype is float32

        output_tensor = model(tensor); print('output_tensor shape', output_tensor.shape)
        # convert the output tensor to a numpy array
        output_arr = output_tensor.squeeze().numpy(); print('output_arr shape', output_arr.shape)
        # remap the output to the original temperature range
        output = remap(output_arr, curr_range=(0,1), new_range=(min_temp, max_temp), to_uint8=False)
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
        
# stop capture and quit
mi48.stop()
cv2.destroyAllWindows()