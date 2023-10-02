# Generative-Models-for-FSS-Inverse-Design
![python_verion](https://user-images.githubusercontent.com/87009163/168636733-f6c303bd-6f3f-4215-b008-c37ea05c0921.svg)

PyTorch implementation of the paper "A Solution to the Dilemma for FSS Inverse Design Using Generative Models".

Our paper is under review by *IEEE Transactions on Antennas and Propagation*.

Updates: Our paper has been accepted by *IEEE Transactions on Antennas and Propagation*. [2023/04/01]

## Table of Contents

- [Background](#background)
- [Case of FSS Inverse Design](#case-of-fss-inverse-design)
- [Get Started](#get-started)
- [Generative models: Collaborating with a well-trained inverse network](#generative-models-collaborating-with-a-well-trained-inverse-network)
	- [Fourier layer](#fourier-layer)
- [Generative models: An end-to-end paradigm](#generative-models-an-end-to-end-paradigm)
- [Contact us](#contact-us)

## Background
Non-unique mapping of data is a huge challenge when using traditional discriminative neural networks to reverse engineer FSS. Existing methods for improving discriminative neural networks have limitations because they do not fundamentally solve this problem. We analyze this existing dilemma from the perspective of information loss caused by data dimensionality reduction, and propose deploying generative models as a solution, for the first time.

## Case of FSS Inverse Design
<table align="center" bgcolor="white">
    <tr>
        <td align="center"><img id="Case" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Case.jpg" width="320" height="200" alt="Case"/><br><span>(a)</span></td>
        <td align="center"><img id="Result" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Result.jpg" width="320" height="200" alt="Result"/><br><span>(b)</span></td></td>
    </tr>
    <tr>
        <td align="center"><img id="Measurement" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Measurement.jpg" width="320" height="200" alt="Measurement"/><br><span>(c)</span></td></td>
        <td align="center"><img id="Verification" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Verification.jpg" width="320" height="200" alt="Verification"/><br><span>(d)</span></td></td>
    </tr>
</table>

Fig. (a) presents an example of an FSS inverse design, where the electromagnetic performance of the FSS can be controlled by tuning four physical parameters.
> In our paper, two approaches with a novel model based on conditional Generative Adversarial Network (cGAN) are presented to achieve inverse design from the given indexes to FSS physical dimensions.

Fig. (b) is a three-dimensional diagram of an actual processed FSS structure, and its physical parameters are obtained by our inverse design method under the condition that the required passband is 6.0~11.0 GHz.

Fig. (c) shows the measurement of the fabricated FSS in an anechoic chamber.

Fig. (d) presents a comparison of the measured and simulated transmission coefficients when the incident wave angle is 0.

## Get Started
Clone the project and install requirments.
> Actually if you just want to run **python** code, the basic **pytorch** package will suffice. Most packages should be version compatible. If you encounter problems, please refer to the version used by the author. 

```sh
    git clone https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design.git
    cd Generative-Models-for-FSS-Inverse-Design
    pip install -r requirements.txt
```
## Generative models: Collaborating with a well-trained inverse network

Please read file `TrainMyCFourierGAN.py` for our implementation details on building and training a cGAN.

### Fourier layer

Please read file `FourierLayer.py` for the implementation details of the *Fourier Layer*.

File `CalFourierLayerGrad.py` is the computation and verification of the gradient of the *Fourier Layer*, which computes the corresponding *Jacobian matrix*.

## Generative models: An end-to-end paradigm

Its implementation please refer to document `TrainMyCFourierGAN.py`. It is worth mentioning that this method is simpler because it does not require the assistance of Fourier layers and inverse networks. More details can be found in our paper.

## Contact us

This repository exposes the core code of the method proposed in our paper. However, considering that the paper has not yet been published, we have not released the relevant dataset for the time being. If you want to know more about our research work, please feel free to contact our email `guzheming@zju.edu.cn`  .
