# %%
# predefinition
import time
import numpy as np
import torch
import difftrace as dg
# load lens
device = torch.device('cpu')
dtype = torch.float64
lens = dg.System('lens_file/doubleGauss.json', torch.float64, torch.device('cpu'))
# define analysis
views = torch.tensor([0., 10., 14.], dtype=dtype, device=device)
wavelengths = torch.tensor([450.00e-6, 500.00e-6, 550.00e-6, 600.00e-6, 650.00e-6], dtype=dtype, device=device)
ana = dg.Analysis(lens, views, wavelengths, dtype=dtype, device=device)
# %%
# test plot setup 2d
ana.plot_setup_2d()
# %% 
# test plot setup 2d with the ray tracing
ana.plot_setup_2d_with_trace()
# %%
# test spot diagram 
ana.spot_diagram()
# %%
# report single ray tracing results
ana.single_ray_trace()
# %%
# wavefront map of the system
_ = ana.wavefront_map()
# %%
# different psf calculation method
import matplotlib.pyplot as plt
pupil_sampling = 201
image_sampling = 101
image_delta = 0.0002
sample_distribution = 'hexapolar'
psf_spot = ana.psf_spot(pupil_sampling=pupil_sampling, 
                        image_sampling=image_sampling,
                        image_delta=image_delta)
plt.imshow(psf_spot, cmap='jet')
# %%
psf_coherent = ana.psf_coherent(pupil_sampling=pupil_sampling, 
                                image_sampling=image_sampling,
                                image_delta=image_delta)
plt.imshow(psf_coherent, cmap='jet')
# %%
psf_huygens = ana.psf_huygens(pupil_sampling=pupil_sampling, 
                                image_sampling=image_sampling,
                                image_delta=image_delta)
plt.imshow(psf_huygens, cmap='jet')
# %%
psf_kirchoff = ana.psf_kirchoff(pupil_sampling=pupil_sampling, 
                                image_sampling=image_sampling,
                                image_delta=image_delta)
plt.imshow(psf_kirchoff, cmap='jet')
# %%
# different mtf calculation method
_, _, _ = ana.mtf(pupil_sampling, image_sampling, image_delta, method='coherent', show=True)
# %%
_, _, _ = ana.mtf(pupil_sampling, image_sampling, image_delta, method='kirchoff', show=True)
# %%
# mtf through focus
MTF_T_through_focus, MTF_S_through_focus = ana.mtf_huygens_through_focus(pupil_sampling=100, 
                                                                         image_sampling=100, 
                                                                         image_delta=0.0002, 
                                                                         frequency=25,
                                                                         delta_focus=0.01, 
                                                                         steps=11)
plt.plot(MTF_T_through_focus)
plt.plot(MTF_S_through_focus)
# %%
# relative illumination
field_density = 11
ri_list = ana.relative_illumination(field_density)
plt.plot(np.asarray(ri_list))
# %%
