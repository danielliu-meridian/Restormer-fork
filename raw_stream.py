"""
Simple script to show the raw MI48 output in grayscale. We are interested in improving the grayscale
image produced by MI48. 
"""

import os
import sys
import time
from senxor.utils import data_to_frame, remap, connect_senxor, get_default_outfile
# from senxor.display import cv_render
import cv2 as cv
import numpy as np
import torch
from runpy import run_path


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

mi48 = connect_senxor()
ncols, nrows = mi48.fpa_shape

mi48.regwrite(0xD0, 0x00)
mi48.regwrite(0x20, 0x00)
mi48.regwrite(0x25, 0x00)
mi48.regwrite(0x30, 0x00)
mi48.regwrite(0xB4, 0x01)
mi48.regwrite(0x02, 0x00)

mi48.set_emissivity(0.95)
mi48.set_sens_factor(1.0)
mi48.set_offset_corr(0.0)
mi48.set_otf(0.0)

# initiate continuous frame acquisition
time.sleep(1)
with_header = True
mi48.start(stream=True, with_header=with_header)

scale = 2
hflip = False

data, header = mi48.read()      # check that mi48 can be read
if data is None:
    mi48.stop()
    sys.exit(1)
else:
    #####   Video path set up
    videoroot = "C:/Users/takao/Desktop/denoising_raw_data"
    # savefile = get_default_outfile(ext="avi")
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    frameSize = (int(ncols * scale),  int(nrows * scale))
    out_raw = cv.VideoWriter(os.path.join(videoroot, "raw_output4.avi"), fourcc, 12.0, frameSize)
    #####

frame_lst = []
frameidx = 0

while True:
    data, header = mi48.read()
    if data is None:
        mi48.stop()
        sys.exit(1)

    frameidx += 1
    frame = data_to_frame(data, (ncols, nrows), hflip=hflip)
    frame = cv.resize(remap(frame).astype(np.uint8), dsize=None,
                      fx=scale, fy=scale, interpolation=cv.INTER_CUBIC).astype(np.uint8)
    img = np.dstack((frame, frame, frame))
    if  30 < frameidx <= 240:
        frame_lst.append(frame.copy())
        out_raw.write(img)
        print(f"saved frame {frameidx}")
    
    cv.imshow("",img)
    key = cv.waitKey(1)  # & 0xFF
    if key == ord("q"):
        break

mi48.stop()
out_raw.release()
cv.destroyAllWindows()
sys.exit(0)

processed_lst = []
out_processed = cv.VideoWriter(os.path.join(videoroot, "processed_output1.avi"), fourcc, 12.0, frameSize)

for frame in frame_lst:
    input_ = np.expand_dims(frame, axis=2)
    print(f"input_.shape: {input_.shape}")
    input_ = torch.from_numpy(input_).float().div(255.).permute(2,0,1).unsqueeze(0)
    output_ = remap(model(input_).squeeze(0).squeeze(0).detach().numpy())

    output_ = np.dstack((output_, output_, output_))
    out_processed.write(output_)
    
    out_processed.release()

print("DONE")
