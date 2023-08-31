# KIDD
Implementations of 'Kernel Ridge Regression-Based Graph Dataset Distillation', SIGKDD'23.

The implementation of the LiteGNTK is adapted from the GNTK implmentation (https://github.com/KangchengHou/gntk).

If you have any questions about this repo, feel free to drop an email to zhexu3@illinois.edu

## Some important notes
1. The LiteGNTK is implemented in **PyTorch** so that we use the autogradient for updating the synthetic graphs.
2. Both the original graphs and synthetic graphs are presented as **3-d tensors** (via paddings and truncations). To be specific, the adjacency matrices are presented as (# graphs, # nodes, # nodes), and node features are presented as (# graphs, # nodes, # features).  Based on this presentation, the LiteGNTK computation is implemented by fast tensor computation (e.g., torch.einsum), which is much faster than matrix multiplication in loops.
3. In the cost of fast computation, the synthetic graphs are **of the same size**, which is a potential defect of this implementation. Masking a part of the entries in the adjacency tensor and node feature tensor may work, but we leave it as future works.

## Commands

The command to reproduce the results can be found in commands.sh, where the parameter "gpc" denotes the number of graphs per class to be synthesized.

## Citation

If you find this repository useful, please kindly cite the following paper.

```
@inproceedings{DBLP:conf/kdd/XuCPCDYT23,
  author       = {Zhe Xu and
                  Yuzhong Chen and
                  Menghai Pan and
                  Huiyuan Chen and
                  Mahashweta Das and
                  Hao Yang and
                  Hanghang Tong},
  editor       = {Ambuj Singh and
                  Yizhou Sun and
                  Leman Akoglu and
                  Dimitrios Gunopulos and
                  Xifeng Yan and
                  Ravi Kumar and
                  Fatma Ozcan and
                  Jieping Ye},
  title        = {Kernel Ridge Regression-Based Graph Dataset Distillation},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6-10, 2023},
  pages        = {2850--2861},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3580305.3599398},
  doi          = {10.1145/3580305.3599398},
  timestamp    = {Fri, 18 Aug 2023 08:45:08 +0200},
  biburl       = {https://dblp.org/rec/conf/kdd/XuCPCDYT23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```