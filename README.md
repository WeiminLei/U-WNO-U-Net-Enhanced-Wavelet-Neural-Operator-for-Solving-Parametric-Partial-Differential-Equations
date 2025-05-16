# U-Net Enhanced Wavelet Neural Operator (U-WNO)
This repository provides the Python code of the numerical examples in the paper "U-WNO: U-Net Enhanced Wavelet Neural Operator for Solving Parametric Partial Differential Equations".

## Files
A brief description of these files is provided below.
```
  + `uwno1d_Advection.py`: For wave advection equation (time-independent problem).
  + `uwno1d_Advection_time.py`: For wave advection equation (time-dependent problem).
  + `uwno1d_Burger_time.py`: For Burgers' equation with discontinuous field (time-dependent problem).
  + `uwno1d_Burgers.py`: For Burger's equation (time-independent problem).
  + `uwno2d_AC.py`: For Allen-Cahn reaction-diffusion equation (time-independent problem).
  + `uwno2d_Darcy_notch.py`: For Darcy equation with a notch in triangular domain (time-independent problem).
  + `uwno2d_Darcy.py`: For Darcy equation in a rectangular field (time-independent problem).
  + `uwno2d_NS.py`: For Navier-Stokes equation (time-dependent problem).
  + `uwno2d_possion.py`: For Non-homogeneous Poisson equation (time-independent problem)
  + `utils.py` contains some useful functions for data handling (improvised from [FNO paper](https://github.com/zongyi-li/fourier_neural_operator)).
  + `wavelet_convolution.py` contains functions for 1D, 2D, and 3D convolution in wavelet domain (improvised from [WNO paper](https://github.com/csccmiittd/WNO)).
```

## The required Python library
To run the above code, the following software packages need to be installed:
  + [PyTorch](https://pytorch.org/)
  + [PyWavelets - Wavelet Transforms in Python](https://pywavelets.readthedocs.io/en/latest/)
  + [Wavelet Transforms in Pytorch](https://github.com/fbcotter/pytorch_wavelets)
  + [Wavelet Transform Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)

Copy all the data in the 'data' folder and place the 'Data' folder in the parent folder where the code is located. If the location of the data changes, the path of the data should be given in the code.

## Testing
To perform predictions on the new input, you can use 'UWNO_testing_(.) .py 'code. The training model for generating results for the U-WNO paper can be found in the following link:
  > [Models](https://1drv.ms/f/s!Alcbal0ytZ4dkWp36Nf6GZhbJtGK)

## Dataset
  + The training and testing datasets for the (1) Burgers equation with discontinuity in the solution field (section 4.1), and (2) Allen-Cahn equation (section 4.4) are available in the following link:
    > [Dataset-1](https://drive.google.com/drive/folders/1scfrpChQ1wqFu8VAyieoSrdgHYCbrT6T?usp=sharing)
  + The datasets for (1) Burgers equation ('burgers_data_R10.zip')(section 4.1), (2) Darcy flow equation in a rectangular domain ('Darcy_421.zip')(section 4.3), (3) 2-D time-dependent Navier-Stokes equation ('ns_V1e-3_N5000_T50.zip')(section 4.6), are taken from the following link:
    > [Dataset-2](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
  + The datasets for Darcy flow equation with a notch in triangular domain ('Darcy_Triangular_FNO.mat')(section 4.3) and time-dependent wave advection equation(section 4.2) are taken from the following link:
    > [Dataset-3](https://github.com/lu-group/deeponet-fno/tree/main/data)
  + The datasets for Non-homogeneous Poisson equation (section 4.5) are taken from the following link:
    >[Dataset-4](https://1drv.ms/f/s!Alcbal0ytZ4dkWp36Nf6GZhbJtGK)
    
