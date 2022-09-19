# Generative-Models-for-FSS-Inverse-Design
![python_verion](https://user-images.githubusercontent.com/87009163/168636733-f6c303bd-6f3f-4215-b008-c37ea05c0921.svg)

PyTorch implementation of the paper "A Solution to the Dilemma for FSS Inverse Design Using Generative Models".

Our paper is under review by *IEEE Transactions on Antennas and Propagation*.

## Table of Contents

- [Background](#background)
- [Case of FSS Inverse Design](#case-of-fss-inverse-design)
- [Get Started](#get-started)

## Background
Non-unique mapping of data is a huge challenge when using traditional discriminative neural networks to reverse engineer FSS. Existing methods for improving discriminative neural networks have limitations because they do not fundamentally solve this problem. We analyze this existing dilemma from the perspective of information loss caused by data dimensionality reduction, and propose deploying generative models as a solution, for the first time.

## Case of FSS Inverse Design
<table align="center">
    <tr>
        <td><img id="Case" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Case.jpg" width="320" height="200" alt="demo"/></td>
        <td><img id="Result" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Result.jpg" width="320" height="200" alt="demo"/></td>
    </tr>
    <tr>
        <td><img id="Measurement" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Measurement.jpg" width="320" height="200" alt="demo"/></td>
        <td><img id="Verification" src="https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design/blob/main/Img/Verification.jpg" width="320" height="200" alt="demo"/></td>
    </tr>
</table>

> In this case, two approaches with a novel model based on conditional Generative Adversarial Network (cGAN) are presented to achieve inverse design from the given indexes to FSS physical dimensions.

## Get Started
Clone the project and install requirments.
> Actually if you just want to run **python** code, the basic **pytorch** package will suffice. Most packages should be version compatible. If you encounter problems, please refer to the version used by the author. 

```sh
    git clone https://github.com/ushione/Generative-Models-for-FSS-Inverse-Design.git
    cd Generative-Models-for-FSS-Inverse-Design
    pip install -r requirements.txt
```

