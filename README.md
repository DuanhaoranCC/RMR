# RMR
This repo is for the source code of the KDD 2024 paper "Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation."

## Dependencies

```python
pip install -r requirements.txt
```

## Compute Resources
All our experiments are performed on a computer with Intel(R) Core(TM) i9-11900K @ 3.50GHz CPU and NVIDIA RTX3090, RTX A6000. The operating system of the computer is Ubuntu 20.04.4 LTS.


## Dataset
We provide datasets for two **Small Heterogeneous Graph** ACM, IMDB, and three **Large Heterogeneous Graph** Aminer PubMed DBLP.

PubMed has a too-large size, We provide the PubMed dataset at [https://drive.google.com/file/d/126IXw-dJGGS9VtjkoUbc6lHUYCblPlJu/view?usp=sharing](https://drive.google.com/drive/folders/1aPX6VvRYbemhHKfVtr0ktq1wLAMaorjr?usp=drive_link).
DBLP is Citation-network V1 at https://www.aminer.cn/citation.

### Explanation of the PubMed Dataset

We hope to see more methods tested on the **PubMed dataset**. (Based PGB: A PubMed Graph Benchmark for Heterogeneous Network Representation Learning, CIKM2023)

1. **Data Splitting**:
   - The dataset is divided by **year**:
     - Papers published **before 2018** are used as the **training set**.
     - Papers published **between 2018 and 2019** are used as the **validation set**.
     - Papers published **after 2019** are used as the **test set**.

2. **Task Type**:
   - The task is a **multi-class classification** task, as each article can have multiple **MeSH** keywords.

3. **Task Difficulty**:
   - The task difficulty is considered **high**, as indicated by a low **F1 score** of approximately **30%**.

You can run the build_pubmed function at load.py.

## Training model 

You can modify the code to specify which dataset you wish to run.

```python
python train.py
```

We provide model weights of Aminer dataset at https://drive.google.com/file/d/126IXw-dJGGS9VtjkoUbc6lHUYCblPlJu/view?usp=sharing.

## Cite
```
@inproceedings{RMR,
  author    = {Haoran Duan, Cheng Xie, and Linyu Li},
  title     = {Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation},
  booktitle = {{KDD} '24: The 30th {ACM} {SIGKDD} Conference on Knowledge Discovery
               and Data Mining, Barcelona, Spain, August 25-29, 2024},
  pages     = {689-700},
  year      = {2024}
}
```
