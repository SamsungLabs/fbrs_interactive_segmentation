import os
import torch
import numpy as np
from torch._C import dtype
from interactive_demo import App_v2
from PIL import Image
from isegm.inference import utils

class fbrs_engine():
    def __init__(self,checkpoint, norm_radius=260):

        # automatically check if device supports cuda
        if torch.cuda.is_available():
            print('CUDA detected!')
            device = torch.device("cuda")        
        else:
            print('CUDA not detected! Switching to CPU instead.')
            device = torch.device("cpu")

        # find location of where checkpoint file is stored
        module_path =  os.path.dirname(os.path.abspath(__file__))
        INTERACTIVE_MODELS_PATH = os.path.join(module_path,'weights')

        torch.backends.cudnn.deterministic = True
        checkpoint_path = utils.find_checkpoint(
            INTERACTIVE_MODELS_PATH, checkpoint)
        model = utils.load_is_model(
            checkpoint_path, device, cpu_dist_maps=True, norm_radius=norm_radius)
        
        self.model = model
        self.device = device


    def predict(self, x_coords, y_coords, is_pos, image, threshold=0.5, save_path=None, limit_longest_size=800,brs_mode='f-BRS-B'):
        # brs_mode options: ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']

        predictor_params={'brs_mode': brs_mode}
        app_class = App_v2.AppReplacement(image, threshold, self.model, x_coords, y_coords, is_pos, self.device,
                                          limit_longest_size=limit_longest_size,predictor_params=predictor_params)
        mask = app_class.prediction

        if mask is None:
            mask = np.zeros((image.shape[1],image.shape[0]))

        binary_pred = mask > threshold  # threshold typically set to 0.5
        mask_pred = binary_pred.astype('int') * 255
        mask_pred = mask_pred.astype('uint8')

        if save_path is not None:
            pil_img = Image.fromarray(mask_pred)
            pil_img.save(save_path)

        return mask_pred

if __name__ == '__main__':
    # unit test (blank image)
    checkpoint = 'resnet34_dh128_sbd' 
    engine = fbrs_engine(checkpoint)
    x_coord = []
    y_coord = []
    is_pos = []
    image = np.zeros((500,500,3),dtype=np.uint8)
    mask_pred = engine.predict(x_coord, y_coord, is_pos, image)
    print('test passed!')