# LexSub:Learning Lexical Relations in a Distributional Vector Space

This repository is the official implementation of [LexSub: Learning Lexical Relations in a Distributional Vector Space](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00316).



 ![LexSub vs Retrofitting Approaches ](img/Concept.png "LexSub Approach" ) 

LexSub is a framework to unify lexical and distributional semantics by defining lexical subspaces of the distributional vector space (word embeddings) in which a lexical relation should hold. LexSub can handle symmetric attract and repel relations (e.g., synonymy and antonymy, respectively), as well as asymmetric relations (e.g., hypernymy and meronomy).
 
 ![LexSub Subspaces ](img/subspaces_combined.png "LexSub Subspaces" ) 


## Pre-trained Models
You can download post-hoc retrofitted embeddings from here:
* [glove-6B-300d](http://)

We will release new embeddings soon.
## Citation
```
@article{arora2020learning,
  title={Learning Lexical Subspaces in a Distributional Vector Space},
  author={Arora, Kushal and Chakraborty, Aishik and Cheung, Jackie CK},
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={311--329},
  year={2020},
  publisher={MIT Press}
}
```

## Training LexSub Embeddings from Scratch:

## Installation Instructions:

### Requirements:
* python3
* pytorch==0.4.1
* GloVe 6B.300d embeddings.

Clone the repository and then run
```
pip install -r requirements.txt
```
Create the data directory and download GloVe embeddings.
```
mkdir -p data/glove; cd data/glove;
wget http://nlp.stanford.edu/data/glove.6B.zip .
unzip glove.6B.zip;
cd ../../;
```

Preprocessing to annotate GloVe vocab with their WordNet relations:
```
python preprocessing/main.py --model retro --lower --data data/glove/
```

## Training LexSub model:
```
output_dir_prefix=output/syn_hyp_mer_0.01_0.01_0.001_allennlp_original epoch=100 synr=0.01 hypr=0.01 merr=0.001 syn=true hyp=true mer=true syn_ratio=${synr} hyp_ratio=${hypr} mer_ratio=${merr} data=glove mdl=retro n_margin=0.5 neg_wn_ratio=10 lr=0.5 ./scripts/run_once.sh 

```


## Results
 ![Similarity and Relatedness Results ](img/similarity_relatedness.png "Similarity and Relatedness Results" ) 

 ![Hyerpnymy Results ](img/hypernymy_evaluation.png "Hypernymy Results" ) 

 ![Extrinsic Task Results ](img/extrinsic_results.png "Extrinsic Task Results" ) 

## Contributing
* If you find a bug or want to propose an improvement, please open a github issue.
* If you would like to submit an improvement, bug fix or want to list new LexSub embeddings on this page (w/ acknowledgement), please submit a pull request.
