# QAIE-ABSA
QAIE-ABSA
# [QAIE: LLM-based Quantity Augmentation and Information Enhancement for few-shot Aspect-Based Sentiment Analysis](https://www.sciencedirect.com/science/article/pii/S0306457324002760?dgcid=author)

## Installation

Create a conda environment with pytorch and scikit-learn :

```
conda create --name QAIE python=3.8
source activate QAIE
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment

```
python setup.py install
```

## Datasets

Once the datasets are downloaded, go to the tkbc/ folder and add them to the package data folder by running :

```
python process_icews.py
python process_timegran.py --tr 100 --dataset yago11k
python process_timegran.py --tr 1 --dataset wikidata12k
# For wikidata11k and yago12k, change the tr for changing time granularity
```

This will create the files required to compute the filtered metrics.

## Reproducing results of TeMME

In order to reproduce the results of TeMME on the four datasets in the paper, go to the tkbc/ folder and run the following commands

```
python learner.py --dataset ICEWS14 --model TeLM --rank 2000 --emb_reg 0.0075 --time_reg 0.01 

python learner.py --dataset ICEWS05-15 --model TeLM --rank 2000 --emb_reg 0.0025 --time_reg 0.1

python learner.py --dataset yago11k --model TeLM --rank 2000 --emb_reg 0.025 --time_reg 0.001

python learner.py --dataset wikidata12k --model TeLM --rank 2000 --emb_reg 0.025 --time_reg 0.0025

```


## Acknowledgement

We refer to the code of QAIE. Thanks for their great contributions!

## Cite Information
Lu H, Liu T, Cong R, et al. QAIE: LLM-based Quantity Augmentation and Information Enhancement for few-shot Aspect-Based Sentiment Analysis[J]. Information Processing & Management, 2025, 62(1): 103917.

```
@article{lu2025qaie,
  title={QAIE: LLM-based Quantity Augmentation and Information Enhancement for few-shot Aspect-Based Sentiment Analysis},
  author={Lu, Heng-yang and Liu, Tian-ci and Cong, Rui and Yang, Jun and Gan, Qiang and Fang, Wei and Wu, Xiao-jun},
  journal={Information Processing \& Management},
  volume={62},
  number={1},
  pages={103917},
  year={2025},
  publisher={Elsevier}
}
```


Contact: liutianci@stu.jiangnan.edu.cn


