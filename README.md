## Rotate3D: Representing Relations as Rotations in Three-Dimensional Space for Knowledge Graph Embedding 

This is the code of [Rotate3D](https://dl.acm.org/doi/abs/10.1145/3340531.3411889).  

### Dependencies

- Python 3.7+
- [PyTorch](http://pytorch.org/) 1.0+

### The implementation of Rotate3D

![](https://latex.codecogs.com/svg.latex?\begin{aligned}&\mathbf{v}_{\parallel}%20=%20(\mathbf{v}\cdot\mathbf{u})\mathbf{u}%20\\&\mathbf{v}_{\perp}%20=%20\mathbf{v}%20-%20\mathbf{v}_{\parallel}%20=%20\mathbf{v}%20-%20(\mathbf{v}\cdot\mathbf{u})\mathbf{u}%20\\&\mathbf{w}=\mathbf{u}\times\mathbf{v}_{\perp}=\mathbf{u}\times\mathbf{v}%20\\&%20qvq^{-1}=qvq^*%20\\%20=%20&%20[\cos\frac{\theta}{2},%20\sin\frac{\theta}{2}\mathbf{u}][0,%20\mathbf{v}][\cos\frac{\theta}{2},%20-\sin\frac{\theta}{2}\mathbf{u}]%20\\%20=&[0,%20\mathbf{v}_{\parallel}+\cos{\theta}\mathbf{v}_{\perp}+\sin{\theta}\mathbf{w}]%20\\%20=&%20[0,\cos{\theta}\mathbf{v}+%20(1-\cos{\theta})(\mathbf{v}\cdot\mathbf{u})\mathbf{u}+\sin{\theta}(\mathbf{u}\times\mathbf{v})]\end{aligned})

We use the expression $\cos{\theta}\mathbf{v}+ (1-\cos{\theta})(\mathbf{v}\cdot\mathbf{u})\mathbf{u}+\sin{\theta}(\mathbf{u}\times\mathbf{v})$ to implement Rotate3D. 

### Link Prediction

To reproduce the results of Rotate3D, run the following commands.

```
cd LP

# WN18
bash runs.sh train Rotate3D wn18 0 0 512 256 1000 12.0 1.0 0.0001 80000 8 0 2 --disable_adv 

# FB15k
bash runs.sh train Rotate3D FB15k 0 0 1024 256 1000 24.0 0.5 0.00005 150000 8 0 2

# WN18RR
bash runs.sh train Rotate3D wn18rr 0 0 512 256 500 6.0 1.0 0.00005 80000 8 0.1 1 --disable_adv

# FB15k-237
bash runs.sh train Rotate3D FB15k-237 0 0 1024 256 1000 12.0 1.0 0.00005 100000 8 0 2
```

## Citation

If you find this code useful, please consider citing the following paper:

```
@inproceedings{
  gao2020rotate3d,
  title={Rotate3D: Representing Relations as Rotations in Three-Dimensional Space for Knowledge Graph Embedding},
  author={Gao, Chang and Sun, Chengjie and Shan, Lili and Lin, Lei and Wang, Mingjiang},
  booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
  pages={385--394},
  year={2020}
}
```

## Acknowledgement

We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding), [HAKE](https://github.com/MIRALab-USTC/KGE-HAKE). Thanks for their contributions.