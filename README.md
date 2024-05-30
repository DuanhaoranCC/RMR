# RMR
This repo is for the source code of the KDD 2024 paper "Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation."

## Dependencies

```python
pip install -r requirements.txt
```

## Compute Resources
All our experiments are performed on a computer with Intel(R) Core(TM) i9-11900K @ 3.50GHz CPU and NVIDIA RTX3090, RTX A6000. The operating system of the computer is Ubuntu 20.04.4 LTS.


## Dataset
We provide datasets for ACM, IMDB, Aminer, and DBLP. However, the DBLP dataset might not be uploaded to GitHub due to space constraints.

## Training model 

You can modify the code to specify which dataset you wish to run.

```python
python train.py
```

## Cite
```
@inproceedings{RMR,
  author    = {Haoran Duan, Cheng Xie, and Linyu Li},
  title     = {Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation},
  booktitle = {{KDD} '24: The 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Barcelona, Spain, August 25-29, 2024},
  pages     = {--},
  year      = {2024}
}
```
## Contact
If you have any questions, please feel free to contact me with hrduan07@gmail.com
