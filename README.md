# Imposing WordNet relation on Distributional Word Embeddings

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.

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
