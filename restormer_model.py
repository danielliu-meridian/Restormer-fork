import os
import torch
from runpy import run_path

# get weights and parameters for the task
def get_weights_and_parameters(task):
    """
    Get weights and parameters for the task
    Args:
        task (str): task name
        parameters (dict): basic parameters for all tasks
    Returns:
        weights (str): weights path
        parameters (dict): parameters for the task
    """
    # the basic parameters for all tasks
    parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 
              'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 
              'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
    # if task is denise, get the weights and parameters for denoise task
    if task == 'denoise':
        # weights path
        weights = os.path.join('Denoising', 'pretrained_models', 'gaussian_gray_denoising_blind.pth')
        # set the LayerNorm_type to BiasFree
        parameters['LayerNorm_type'] =  'BiasFree'
        return weights, parameters
    else:
        return None, None
    
def load_model(parameters, weights):
    """
    Load the model
    Args:
        path (str): path to the model
        parameters (dict): parameters for the model
        weights (str): weights path
    Returns:
        model: the model
    """
    # load the model
    load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = load_arch['Restormer'](**parameters)
    # set the model to device 
    # if cuda is available, set the device to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # load the weights
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    params = checkpoint['params']
    # load the model state dict
    model.load_state_dict(params)
    # set the model to evaluation mode
    model.eval()
    return model