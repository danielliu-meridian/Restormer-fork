import cv2
import os
import numpy as np
import threading
import queue
import time
import torch
from runpy import run_path

from senxor.utils import remap, data_to_frame, connect_senxor
from senxor.display import cv_render
from senxorplus.stark import STARKFilter
from senxor.filters import RollingAverageFilter


####
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
####


####
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
'lm_atype': 'ra',
'lm_ks': (5,5),
'lm_ad': 9,
'alpha': 2.0,
'beta': 2.0,}
frame_filter = STARKFilter(stark_par)

scale = 2
hflip = False

mask = np.ones((nrows, ncols), dtype=bool)
exit_flag = False
####

# Function to read frames from the camera
def read_frames(mi48, queue):
    while not exit_flag:
        data, _ = mi48.read()

        if data is None:
            mi48.stop()
            break

        frame = data_to_frame(data, (ncols, nrows), hflip=hflip)
        corner_temp = frame[mask].mean()
        frame[mask==False] = corner_temp

        min_temp = minav(np.median(np.sort(frame.flatten())[:16]))
        max_temp = maxav(np.median(np.sort(frame.flatten())[-5:]))
        frame = np.clip(frame, min_temp, max_temp)

        frame = frame_filter(frame)
        min_temp = minav(np.median(np.sort(frame.flatten())[:9]))
        max_temp = maxav(np.median(np.sort(frame.flatten())[-5:]))
        frame = np.clip(frame, min_temp, max_temp)

        if not queue.empty():
            # Clear the queue if it already has a frame to keep only the latest frame
            try:
                _ = queue.get_nowait()
            except queue.Empty:
                pass

        queue.put(frame)


# Function to do inference on frames
def infer_frames(queue):
    while True:
        frame = queue.get()
        # Perform inference on the frame (replace this with your actual inference code)
        with torch.no_grad():
            input_ = np.expand_dims(remap(frame), axis=2)
            input_ = torch.from_numpy(input_).float().div(255.).permute(2,0,1).unsqueeze(0)
        
        output_ = remap(model(input_).squeeze(0).squeeze(0).detach().numpy())
        output_ = np.dstack((output_, output_, output_))
        
        # Display the predicted frame using cv2.imshow
        cv2.imshow("Inference", cv2.resize(output_, dsize=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            global exit_flag
            exit_flag = True
            break

# Create a queue to store frames
frame_queue = queue.Queue()

# Create threads for reading frames and inference
read_thread = threading.Thread(target=read_frames, args=(mi48, frame_queue))
infer_thread = threading.Thread(target=infer_frames, args=(frame_queue,))

# Start the threads
read_thread.start()
infer_thread.start()

# Wait for the threads to finish
read_thread.join()
infer_thread.join()

# Release the VideoCapture and close all OpenCV windows
mi48.stop()
cv2.destroyAllWindows()