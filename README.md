# RMR
This repo is for the source code of the KDD 2024 paper "Reserving-Masking-Reconstruction Model for Self-Supervised Heterogeneous Graph Representation."

## Dependencies

```python
pip install -r requirements.txt
```

## Dataset
We provide datasets for two **Small Heterogeneous Graph** ACM, IMDB, and three **Large Heterogeneous Graph** Aminer PubMed DBLP.

### Explanation of the PubMed Dataset
1. **Data Splitting**:
   - The dataset is divided by **year**:
     - Papers published **before 2018** are used as the **training set**.
     - Papers published **between 2018 and 2019** are used as the **validation set**.
     - Papers published **after 2019** are used as the **test set**.

2. **Task Type**:
   - The task is a **multi-class classification** task, as each article can have multiple **MeSH** keywords.

3. **Task Difficulty**:
   - The task difficulty is considered **high**, as indicated by a low **F1 score** of approximately **30%**.

PubMed has a too-large size, We provide the sampling PubMed dataset at [https://drive.google.com/file/d/126IXw-dJGGS9VtjkoUbc6lHUYCblPlJu/view?usp=sharing](https://drive.google.com/drive/folders/1aPX6VvRYbemhHKfVtr0ktq1wLAMaorjr?usp=drive_link).
You can run the build_pubmed function at load.py.

| Dataset   | # Node | # Edge | HGT          | Random-Init       | RMR          |
|-----------|--------|--------|--------------|-------------------|--------------|
| PubMed v1 | 2.1M   | 7.46M  | 23.17 ± 0.3  | 20.44 ± 0.1       | 29.98 ± 0.2  |
| PubMed v2 | 2.6M   | 13.76M | 23.11 ± 0.1  | 20.46 ± 0.1       | 32.86 ± 0.1  |
| PubMed v3 | 3.3M   | 22.18M | 23.25 ± 0.1  | 20.43 ± 0.1       | 33.84 ± 0.4  |
| PubMed v4 | 4.1M   | 32.71M | 23.24 ± 0.2  | 20.41 ± 0.1       | 33.53 ± 0.1  |

To assess the performance of the proposed method on a large dataset with varying scales, we extracted **PubMed** v1, v2, v3, and v4 from the biomedical literature heterogeneous graph. The above table provides details on the dataset sizes and the performance of the employed methodology. HGT is considered a large-scale supervised representative method and is used as a baseline.
The HGT implementation leverages the official PyG implementation, utilizing 4 attention heads, 2 layers, a learning rate of 0.005, a weight decay of 0.001, and a hidden layer dimension of 64.

The results indicate that RMR outperforms HGT on datasets of different sizes. Notably, RMR shows a performance improvement of 10% compared to random initialization. This observation suggests that leveraging self-supervised tasks enables the model to effectively capture intricate structural semantics. However, it's noteworthy that the performance of RMR on **PubMed**, a vast and diverse graph, is not as pronounced as in the **Aminer** dataset. This discrepancy might be attributed to **PubMed** being a biomedical network with a specific domain context that involves numerous biomedical proprietary names, making it challenging for the model to infer the Medical Subject Headings (MeSH) of papers solely based on knowledge of citations, etc.

### Explanation of the DBLP Dataset
Coming soon

DBLP is Citation-network V1 at https://www.aminer.cn/citation.

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
