import cv2
import yaml

from pipeline import LaMaPipeline

if __name__ == "__main__":
    config_path = 'lama_config.yaml'
    image_path = 'image.png'
    mask_path = 'mask.png'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    pipeline = LaMaPipeline(config)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    # image = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    res = pipeline.infer(image, mask)
    cv2.imwrite('test_res.jpg', res)
    