# D-Flow: solving image inverse problems by differentiating through a flow

Reimplementation in PyTorch of **D-Flow** ([Ben-Hamu et al., ICML 2024](https://arxiv.org/abs/2402.14017)),
applied to **inpainting, deblurring and ×2 super-resolution** on MNIST, CIFAR-10 and STL-10.

The code is inspired by the paper and by the
[minimal implementation by marvinsxtr](https://github.com/marvinsxtr/d-flow).

## Idea

A Flow Matching model learns a velocity field that transports Gaussian noise $z_0$ to images.
For an inverse problem $y = \mathcal{A}(x)$, D-Flow does **not** retrain anything: it optimises
the source noise so that the generated image matches the measurement,

$$z_0^\* = \arg\min_{z_0} \lVert \mathcal{A}(\mathrm{Flow}(z_0)) - y \rVert^2 ,$$

by back-propagating through the ODE solver. Differentiating through the flow implicitly keeps
the solution close to the data manifold learned by the model, with no explicit regulariser.

## Pipeline

1. **Velocity network:** time-conditioned UNet (1.1M parameters for MNIST/CIFAR-10, 7.5M for STL-10).
2. **Flow Matching training** with the conditional optimal-transport path
   $x_t = (1-t)\,x_0 + t\,x_1$ and target velocity $x_1 - x_0$.
3. **Sanity check:** unconditional generation.
4. **D-Flow:** differentiable ODE solver and optimisation of $z_0$ for each degraded image.
5. **Forward operators:** binary mask (inpainting), Gaussian blur (deblurring), ×2 downsampling (super-resolution).
6. **Evaluation:** visual comparisons and mean PSNR, plus a bonus random-pixel inpainting task (30% of pixels observed).

## Results (mean PSNR, higher is better)

| Dataset | Setting | Inpainting | Deblurring | Super-res ×2 | Random inpainting (30%) |
|---|---|---|---|---|---|
| MNIST (28×28) | 20 epochs | 12.4 dB | 13.5 dB | 19.4 dB | 15.2 dB |
| CIFAR-10 (32×32) | 20 epochs | 18.9 dB | 20.3 dB | 22.4 dB | 22.2 dB |
| CIFAR-10 (32×32) | 50 epochs | 18.3 dB | 19.8 dB | 23.5 dB | 21.7 dB |
| STL-10 (96×96) | 10 epochs, 100 D-Flow steps | 17.4 dB | 18.1 dB | 19.1 dB | 18.7 dB |
| STL-10 (96×96) | 10 epochs, 200 D-Flow steps | 17.6 dB | 19.4 dB | 21.2 dB | 20.4 dB |

**Observations**
- Doubling the number of D-Flow optimisation steps on STL-10 improves every task (up to +2.1 dB in super-resolution).
- Training the flow longer on CIFAR-10 (20 → 50 epochs) helps super-resolution but not the other tasks:
  the prior is not the only bottleneck, and PSNR on a small test set is noisy.
- Inpainting of a large central region is the hardest task: the missing content must be invented by the prior,
  and PSNR penalises plausible but different completions.

## Notebooks

| Notebook | Content |
|---|---|
| `notebooks/dflow_mnist.ipynb` | MNIST |
| `notebooks/dflow_cifar10_20epochs.ipynb` | CIFAR-10, flow trained 20 epochs |
| `notebooks/dflow_cifar10_50epochs.ipynb` | CIFAR-10, flow trained 50 epochs |
| `notebooks/dflow_stl10_100steps.ipynb` | STL-10, 100 D-Flow optimisation steps |
| `notebooks/dflow_stl10_200steps.ipynb` | STL-10, 200 D-Flow optimisation steps |

Datasets are downloaded automatically with `torchvision`. A GPU is strongly recommended
(STL-10 training takes about 13 minutes per epoch on a single GPU).
Trained weights are not stored in this repository.

## Stack

Python · PyTorch · torchvision · NumPy · Matplotlib

## Limitations

- PSNR is evaluated on a small set of test images and is sensitive to the selection.
- Small UNets and short training: results are far below the paper, which uses much larger pretrained models.
- No perceptual metric (LPIPS, FID) is reported; PSNR favours blurry reconstructions.
