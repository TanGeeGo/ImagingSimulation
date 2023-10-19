# Optical aberrations Correction in Postprocessing using Imaging Simulation (TOG 2021, PostRec. SIGGRAPH 2022)
## The code releasing is finally approved by our funding agency. Thanks for your waiting!

by Shiqi Chen, Huajun Feng, Dexin Pan, Zhihai Xu, Qi Li, and Yueting Chen

This is the official Pytorch implementation of "**Optical aberrations Correction in Postprocessing using Imaging Simulation**" [[Paper]](https://dl.acm.org/doi/abs/10.1145/3474088)

🚩 **Updating（New Features/Updates）**
- ✅ Oct. 19, 2023. Add the illustration of psf calculation and the script of analysis module in ray tracing.

## First let me introduce you the how to calculate the psf of a given optical systems

### State the lens and start the analysis
```python
import torch
import difftrace as dg
# load the lens
device = torch.device('cpu')
dtype = torch.float64
lens = dg.System('lens_file/doubleGauss.json', torch.float64, torch.device('cpu'))
# define analysis
views = torch.tensor([0., 10., 14.], dtype=dtype, device=device)
wavelengths = torch.tensor([dg.lambda_F, dg.lambda_d, dg.lambda_C], dtype=dtype, device=device)
ana = dg.Analysis(lens, views, wavelengths, dtype=dtype, device=device)
```
### Calculate the psfs
```python
import matplotlib.pyplot as plt
pupil_sampling = 201
image_sampling = 101
image_delta = 0.0005
sample_distribution = 'hexapolar'
psf_kirchoff = ana.psf_kirchoff(pupil_sampling=pupil_sampling, 
                                image_sampling=image_sampling,
                                image_delta=image_delta)
plt.imshow(psf_kirchoff, cmap='jet')
```

*here we optimize the precalculation of the entrace pupil and the rays sampling, so be free to directly use this method*

### We also provide many other analysis such as `spot diagram`, `mtf`, `wavefront map`, ... Please check the ./PSF_generation/ray_tracing/analysis.ipynb for more information

### This repo is still in updating.