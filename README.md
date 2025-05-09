<h1 align="center">
<br>
<a href="https://github.com/leviome/DearGaussianGUI"><img src="https://github.com/leviome/DearGaussianGUI/blob/main/assets/DearGaussianLogo.png" alt="DearGaussian logo">
</h1>
<h4 align="center">A minimal GUI for 3DGS using DearPyGUI framework.</h4>

```
# download original 3DGS repo, which can be skipped if you have it already.
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

git clone https://github.com/leviome/DearGaussianGUI --recursive
cd DearGaussianGUI

# softlink
ln -s /path/to/gaussian-splatting gs

# set enviroment
conda create -n DearGUI python=3.10
conda activate DearGUI
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# install depth rasterization and original simple-knn
pip install ./submodules/diff-gaussian-rasterization
pip install ./gs/submodules/simple-knn

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
```
@article{huang2023sc,
  title={SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes},
  author={Huang, Yi-Hua and Sun, Yang-Tian and Yang, Ziyi and Lyu, Xiaoyang and Cao, Yan-Pei and Qi, Xiaojuan},
  journal={arXiv preprint arXiv:2312.14937},
  year={2023}
}
```
