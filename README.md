# LaMa-simple-infer
**LaMa 简洁推理代码，只保留生成代码** 

输入单张图片及其对应的 mask，输出 inpainting 后的图片，仅支持 CPU 或单卡推理，代码摘自官方实现：[LaMa](https://github.com/advimman/lama)。  
需要下载[模型权重](https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip)，并解压到当前目录，然后运行 `python extract_weight.py` 提取生成器权重，生成 `generator.ckpt` 文件。  
运行 `python infer.py` 进行推理。