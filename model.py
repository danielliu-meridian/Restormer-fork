# Copyright (C) Meridian Innovation Ltd. Hong Kong, 2020. All rights reserved.
# 
# This contains routines to load the neural net model and related utility
# functions.
#
# Note that model(s) are typically large and load slowly. It is therefore
# advantageous to load them in advance of starting processes, and
# certainly outside of starting processes.
# 
# The primary reason for having this module is to facilitate development
# when multiple models may be deployed for various elements of the pipeline
# e.g. filtering/denoising model, face detection model, fever detection model
# etc.
#

import os
import numpy as np
from runpy import run_path

USE_KERAS = False
if USE_KERAS:
    from keras.models import load_model

import cv2 as cv

if USE_KERAS:
    DEFAULT_MODEL = os.path.join('DnCNN_sigma20', 'model_018.hdf5')
else:
    DEFAULT_MODEL = os.path.join('pb_model', 'DnIRB_thermal_sigma25_depth5.pb')

# To change the array to the correct formart
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

KERNEL = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

def get_model(model_dir, model_name):
    """Load the neural net model if existing."""
    if model_name == True:
        # use default; that is: 
        # -m command option without follow up string value
        model_name = DEFAULT_MODEL
    model_name = os.path.join(model_dir, model_name)
    if USE_KERAS:
        model = load_model(model_name, compile=False)
    else:
        model = cv.dnn.readNetFromTensorflow(model_name)
    print('Finished loading trained neural net model')
    return model

def cnn_filter(data, model):
    """
    Run the data through the noise cancellation CNN filter
    ``data`` must be a float normalised to [0, 1].
    """
    if not USE_KERAS:
        x = cv.dnn.blobFromImage(data.astype(np.float32), scalefactor=1,
                                 size=(data.shape[1], data.shape[0]),
                                 mean=0, swapRB=False)
        # model.setInput(x, name='input')
        model.setInput(x)
        # y = model.forward(outputName='subtract_1/sub')
        y = model.forward()
        output = np.squeeze(y, axis=(0,1))
    else:
        x = to_tensor(data.astype(np.float32))
        y = model.predict(x)
        output = from_tensor(y)
    return output

def get_weights_and_parameters(task, parameters):
    """Get the weights and parameters for the task."""
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
    elif task == 'Deraining':
        weights = os.path.join('Deraining', 'pretrained_models', 'deraining.pth')
    elif task == 'Real_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Color_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_color_denoising_blind.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
    elif task == 'Gaussian_Gray_Denoising':
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        parameters['inp_channels'] =  1
        parameters['out_channels'] =  1
        parameters['LayerNorm_type'] =  'BiasFree'
    return weights, parameters

def run_model(model, image, task):
    """Run the model on the image."""

    # Get model weights and parameters
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 
                  'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 
                  'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    weights, parameters = get_weights_and_parameters(task, parameters)
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
    return model(image)
