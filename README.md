# PDRS
Prerequisite-Driven Recommender System
This is my PyTorch implementation for the paper:
> Hengchang Hu, Yiding Ran, Liangming Pan, Min-Yen Kan (2022). Modeling and Leveraging Prerequisite Context in Recommendation. In Proceedings of Workshop on Context-Aware Recommender Systems (CARS@RecSys’22). ACM, New York, NY, USA.

### Environment Requirement

-----

The code has been tested under Python 3.6.9. The required packages are as follows:

- pytorch == 1.3.1
- numpy == 1.18.1
- scipy == 1.3.2
- sklearn == 0.21.3

### Dataset

----

You can download the preprocessed PDRS data from: https://drive.google.com/file/d/11lcDjbmc1WJysfzVqPDce4VGkP_n2Lfr/view?usp=sharing, and copy the dataset into to `data/` folder.

### Run

----

The instruction of commands has been clearly stated in the codes.

```
​```
python run/main_knowledge_mlp.py
​```
```

More baseline running commond examples can be found in `example_cmd.sh`. The model parameters are pre-set in the conf files at `conf/` folder.