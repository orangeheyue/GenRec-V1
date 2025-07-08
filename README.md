# GenRec-V1| Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/orangeheyue/GenRec-V1">
    <img src="images/haitanglogo.png" alt="Logo" width="400" height="200">
  </a>
</div>

## News
This is the offical code for GenRec-V1(海棠):

>**[ACMMM 2025]** Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation
<img src="images/genrec-v1.png" width="900px" height="250px"/>

## 🌟 Key Highlights​
- Introduces a novel FlipInterestDiffusion framework for unbiased multimedia recommendation​
- Achieves state-of-the-art performance on TikTok and Amazon-Sports datasets across multiple metrics​
- Integrates high-order interest learning (GCNModel) and multimodal interest clustering (MultimodalCluster)

## 📊 Experimental Results
| Dataset       | Metric     | MF-BPR | NGCF  | LightGCN | SGL   | NCL   | HCCF  | VBPR  | LGCN-M | MMGCN | GRCN  | LATTICE | CLCRec | MMGCL | SLMRec | BM3   | DiffMM | GenRec |
| ------------- | ---------- | ------ | ----- | -------- | ----- | ----- | ----- | ----- | ------ | ----- | ----- | ------- | ------ | ----- | ------ | ----- | ------ | ------ |
| TikTok        | Recall@20  | 0.0346 | 0.0604| 0.0653   | 0.0603| 0.0658| 0.0662| 0.0380| 0.0679  | 0.0730| 0.0804| 0.0843   | 0.0621  | 0.0799| 0.0845  | 0.0957| 0.1129  | 0.1165  |
| TikTok        | Precision@20| 0.0017 | 0.0030| 0.0033   | 0.0030| 0.0034| 0.0029| 0.0018| 0.0034  | 0.0036| 0.0036| 0.0042   | 0.0032  | 0.0037| 0.0042  | 0.0048| 0.0056  | 0.0058  |
| TikTok        | NDCG@20    | 0.0030 | 0.0238| 0.0282   | 0.0238| 0.0269| 0.0267| 0.0134| 0.0273  | 0.0307| 0.0350| 0.0367   | 0.0264  | 0.0326| 0.0353  | 0.0404| 0.0456  | 0.0492  |
| Amazon-Sports | Recall@20  | 0.0430 | 0.0695| 0.0782   | 0.0779| 0.0765| 0.0779| 0.0582| 0.0705  | 0.0638| 0.0833| 0.0915   | 0.0651  | 0.0875| 0.0829  | 0.0975| 0.1017  | 0.1062  |
| Amazon-Sports | Precision@20| 0.0023 | 0.0037| 0.0042   | 0.0041| 0.0040| 0.0041| 0.0031| 0.0035  | 0.0034| 0.0044| 0.0048   | 0.0035  | 0.0046| 0.0043  | 0.0051| 0.0054  | 0.0056  |
| Amazon-Sports | NDCG@20    | 0.0202 | 0.0318| 0.0369   | 0.0361| 0.0349| 0.0361| 0.0265| 0.0324  | 0.0279| 0.0377| 0.0424   | 0.0301  | 0.0409| 0.0376  | 0.0442| 0.0458  | 0.0478  |


## Enviroment Requirement
- Python >= 3.8
- PyTorch >= 2.0
- torch-geometric >= 2.3.0
- scikit-learn >= 1.2.0
- numpy >= 1.24.0
  
## 📁 Code Structure
```plaintext
├── images/               # Project visualization assets and figures
├── datasets/             # Dataset directory (to be populated)
├── Main.py               # Core execution script for GenRec diffusion model pipeline
├── Model.py              # Model architecture definitions:
│   ├── FlipInterestDiffusion (core model)
│   ├── GCNModel (high-order interest learning)
│   └── Multimodal feature encoders
├── Params.py             # Hyperparameter configuration and path management
├── interest_cluster.py   # Interest processing modules:
│   ├── MultimodalCluster (interest clustering)
│   └── InterestDebiase (unbiased interest correction)
└── README.md             # Project documentation
```

## 🚀 Quick Start​
### Data Preparation​
Download dataset from Google Drive:​ TikTok/Baby/Sports​
The dataset includes:​
Text features (extracted via Sentence-Transformers)​
Image features (extracted via CNN)​
User-item interaction records​
Place the downloaded data folder (e.g., TikTok/) into the datasets/ directory.

### Usage
1. Place the downloaded data (e.g. `TikTok`) into the `Datasets` directory.
2. Execute the following command:  
- `nohup python Main.py --data tiktok`  
- `nohup python Main.py --data Sports`  

### Dataset  
Download from Google Drive: [TikTok/Baby/Sports](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)  
The data comprises text and image features extracted from Sentence-Transformers and CNN.  


## Citation
If you find GenRec-V1 useful in your research, please consider citing our [GenRec-V1].
```bibtex
@inproceedings{genrec2025, 
    title={Flip is Better than Noise: Unbiased Interest Generation for Multimedia Recommendation}, 
    author={[Author Names]}, 
    booktitle={Proceedings of the 33rd ACM International Conference on Multimedia (ACMMM)}, 
    year={2025} 
}
```

## 📧 Contact​
For questions or issues, please contact [orangeai-research@gmail.com, orangeheyue@gmail.com]