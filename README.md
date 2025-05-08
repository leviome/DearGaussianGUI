<h1 align="center"><img src="https://raw.githubusercontent.com/leviome/DearGaussianGUI/assets/DearGaussianLogo.png" alt="DearGaussian logo"></h1>
<h4 align="center">A minimal GUI for 3DGS using DearPyGUI framework.</h4>

---

```
git clone https://github.com/leviome/DearGaussianGUI --recursive
cd DearGaussianGUI
ln -s gaussian-splatting gs

# run GUI
CUDA_VISBILE_DEVICES=0 python main.py --model_path /path/to/scene/
```
![GUI](assets/screenshot.png)

## Acknowledgement

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}
```
```
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```