# Inversion_with_StyleGANXL
Basic codes for inversion in StyleGANXL


Check out the colab notebook. Inversion for Imagenet works best at 512x512 resolution.

General Observations :

Works reasonably well for OOD Natural Images and even shapes but requires a number of iterations.  

Hard for facial images using Imagenet weights 

Hard for cases where main object occupies proprotionally less space to the fore/back ground

Adding codes for more experiments

1 Super Resolution : We follow PULSE/BRGM's method. Harder than naive inversion requires more iterations till convergence

2 Masking : We follow PULSE/BRGM's method. Harder than naive inversion requires more iterations till convergence

3 Colorization : We follow PULSE/BRGM's method. Harder than naive inversion requires more iterations till convergence
