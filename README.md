# PBLUN
Part-of-speech based label update network for aspect sentiment triplet extraction. Yanbo Li, Qing He, Liu Yang. Journal of King Saud University-Computer and Information Sciences.

The code is based on [BDTF-ABSA](https://github.com/HITSZ-HLT/BDTF-ABSA), and thanks them very much.
## Requirements
Note: We suggest downloading the three files contained in bert-base-uncased and placing them in the "bert-base-uncased" folder to save time.

- transformers==4.15.0
- pytorch==2.0.0+cu118
- einops=0.4.0
- torchmetrics==0.7.0
- tntorch==1.0.1
- pytorch-lightning==1.3.5

## Usage
- ### Training
```
python aste_train.py --dataset 14res
```

## Citation
If you use the code in your paper, please kindly star this repo and cite our paper.
