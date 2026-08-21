import torch
import cv2
import numpy as np

from modules import FFCResNetGenerator
from refine import refine_predict
from utils import preprocess_image, to_tensor
from refine import refine_predict

class LaMaPipeline:
    def __init__(self, config):
        self.config = config
        self.model = FFCResNetGenerator(**config["generator"])        
        self.model.load_state_dict(torch.load(config["model_path"]))
        self.model.eval()
        self.model.to(self.config["device"])
        self.model.requires_grad_(False)

    def infer(self, image, mask):
        src_h, src_w = image.shape[:2]
        #image: RGB (h, w, 3) 0-255, mask: (h, w) 0-255
        image = to_tensor(preprocess_image(image, self.config['modulo'])) # (1, 3, h, w)
        mask = to_tensor(preprocess_image(mask, self.config['modulo'])) # (1, 1, h, w)
        
        if self.config["refine"]: # refine
            batch = {
                "image": image,
                "mask": mask,
                "unpad_to_size": (src_h, src_w)
            }
            out = refine_predict(
                batch, self.model, 
                self.config["device"], 
                self.config["modulo"], 
                **self.config["refiner"]
            )
        else: # generate no refine
            masked_image = image * (1 - mask)
            x = torch.cat([masked_image, mask], dim=1).to(self.config["device"]) # (1, 4, h, w)
            with torch.no_grad():
                out = self.model(x)
                out = mask * out + (1 - mask) * image # (1, 3, h, w)

        cur_res = out[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res
