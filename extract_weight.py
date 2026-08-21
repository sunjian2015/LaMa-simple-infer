import torch

ckpt = torch.load('big-lama/model/best.ckpt', weights_only=False)
state_dict=ckpt['state_dict']
generator_state_dict={}
for k, v in state_dict.items():
    if 'generator' in k:
        generator_state_dict[k]=v
torch.save(generator_state_dict, 'generator.ckpt')
