# PSF generation by ray-tracing and coherent superposition

1. compute the full field PSF data and save it to the path ./PSF_data/PSF_info_fld_${field_value}

    **the '${} represents an variable of field value'**

2. transfer the PSF data in excel to matlab file
```
PSF_data_transfer.m
```

3. complete the coherent superposition of PSFs in different wavelength, and reunion the seperate PSFs according to the wave distribution and lens shading
```
PSF_coherent_superposition.m
```
