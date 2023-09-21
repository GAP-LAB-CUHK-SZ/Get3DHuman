# Get3DHuman: Lifting StyleGAN-Human into a 3D Generative Model using Pixel-aligned Reconstruction Priors (ICCV 2023)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## [Project Page](https://x-zhangyang.github.io/2023_Get3DHuman/) | [Paper](https://arxiv.org/abs/2302.01162) | [Code(Coming soon)](https://github.com/X-zhangyang/Get3DHuman/edit/main/README.md) | [Data](https://drive.google.com/file/d/1f6SS0KUgAzVBEtAerfykCBRmooGeTHWj/view?usp=sharing)

This is the official PyTorch implementation of [Get3DHuman]().


## TODO:triangular_flag_on_post:

- [ ] Preprocess datasets
- [ ] Pretrained weights
- [ ] Demo
- [ ] Training Code


## Requirements

Python 3


Note: A GTX 3090 is recommended to run Get3DHuman, make sure enough GPU memory if using other cards.

## Overview of our framework  <br />
<div align="center">
	<img src="https://github.com/X-zhangyang/Get3DHuman/blob/main/paper_figures/pipeline.png"  width="800">
<br />
</div>

## Multi-view images rendered by Blender.  <br />

<div align="center">
	<img src="https://github.com/X-zhangyang/Get3DHuman/blob/main/paper_figures/sup_multi_view.jpg"  width="600">
<br />
</div>

## Applications

### Interpolation  <br />

<div align="center">
	<img src="https://github.com/X-zhangyang/Get3DHuman/blob/main/paper_figures/sup_inter_0.jpg"  width="600">
<br />
</div>

### Re-texturing  <br />

<div align="center">
	<img src="https://github.com/X-zhangyang/Get3DHuman/blob/main/paper_figures/sup_recolor_all.jpg"  width="600">
<br />
</div>

### Inversion   <br />

<div align="center">
	<img src="https://github.com/X-zhangyang/Get3DHuman/blob/main/paper_figures/inversion.png"  width="800">
<br />
</div>

## Rendering methods   <br />

<div align="center">
	<img src="https://github.com/X-zhangyang/Get3DHuman/blob/main/paper_figures/render_method.png"  width="800">
<br />
</div>

 
## Citation 
If you use Get3DHuman in your research, please consider the following BibTeX entry and give a star🌟!

```bibtex
@inproceedings{xiong2023Get3DHuman,
  author = {Zhangyang Xiong and Di Kang and Derong Jin and Weikai Chen and Linchao Bao and Shuguang Cui and Xiaoguang Han},  
  title = {Get3DHuman: Lifting StyleGAN-Human into a 3D Generative Model using Pixel-aligned Reconstruction Priors},
  booktitle={ICCV},
  year = {2023},
}
```

##  Acknowledgements

Here are some great resources we benefit or utilize from:

- [Stylegan-Human](https://github.com/stylegan-human/StyleGAN-Human)
- [PIFu](https://github.com/shunsukesaito/PIFu)
- [Stylegan2](https://github.com/NVlabs/stylegan2)
