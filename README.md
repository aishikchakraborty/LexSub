# LexSub:Learning Lexical Relations in a Distributional Vector Space

This repository is the official implementation of [LexSub:Learning Lexical Relations in a Distributional Vector Space](https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00316).


LexSub is a framework to unify lexical and distributional semantics by defining lexical subspaces of the distributional vector space (word embeddings) in which a lexical relation should hold. LexSub can handle symmetric attract and repel relations (e.g., synonymy and antonymy, respectively), as well as asymmetric relations (e.g., hypernymy and meronomy).


## Requirements

## Training

## Evaluation

## Pre-trained Models

## Results

## Contributing

```bash
cd preprocessing/
python main.py # Do preprocessing

cd ../
python main.py --cuda --emsize 300 --nhid 700 --dropout 0.5 --mdl Vanilla        #  Train Vanilla LM
python main.py --cuda --emsize 300 --nhid 700 --dropout 0.5 --mdl WN        #  Train Augmented LM
```

## Evaluations
### Word Similarity Task
```bash
cd analogy_tasks/
bash eval.sh
```

### Extrinsic Tasks

```bash
cd extrinsic_tasks/
allennlp train <model_config> -s <output_log_dir>
```

Commands to run:
Glove Retrofitting: 
synr=0.01 hypr=0.01 merr=0.01 output_dir=output/syn_hyp_mer_${synr}_${hypr}_${merr} syn=true hyp=true mer=true syn_ratio=${synr} hyp_ratio=${hypr} mer_ratio=${merr} data=glove mdl=retro ./scripts/run_once.sh 

Glove Original:
mdl=retro emb_text=true output_dir=output/glove emb_filename=glove.6B.300d step=2 ./scripts/run_once.sh

Retrofitting Original:
mdl=retro emb_text=true output_dir=output/retrofitting_original emb_filename=retrofitting_wordnet+_glove.6B.300d step=2 ./scripts/run_once.sh

Retrofitting syn hyp mer:
mdl=retro emb_text=true output_dir=output/retrofitting_syn_hyp_mer emb_filename=retrofitting_syn_hyp_mer_glove.6B.300d step=2 ./scripts/run_once.sh

Counterfitting Original:
mdl=retro emb_text=true output_dir=output/counterfitting_original emb_filename=counterfitting_original_glove.6B.300d step=2 ./scripts/run_once.sh

Counterfitting Syn Ant:
mdl=retro emb_text=true output_dir=output/counterfitting_syn_ant emb_filename=counterfitting_syn_ant_glove.6B.300d step=2 ./scripts/run_once.sh

Counterfitting Syn Ant Hyp Mer:
mdl=retro emb_text=true output_dir=output/counterfitting_syn_ant_hyp_mer emb_filename=counterfitting_syn_ant_hyp_mer_glove.6B.300d step=2 ./scripts/run_once.sh

LEAR Original:
mdl=retro emb_text=true output_dir=output/lear_original emb_filename=lear_original_glove.6B.300d step=2 ./scripts/run_once.sh

LEAR Syn Ant Hyp Mer:
mdl=retro emb_text=true output_dir=output/lear_syn_ant_hyp_mer emb_filename=lear_syn_ant_hyp_mer_glove.6B.300d step=2 ./scripts/run_once.sh
