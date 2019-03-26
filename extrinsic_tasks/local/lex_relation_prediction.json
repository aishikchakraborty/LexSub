{
  "dataset_reader": {
    "type": "lex_rel_classification"
  },
  "train_data_path": "extrinsic_tasks/datasets/lex_rel_train.txt",
  "validation_data_path": "extrinsic_tasks/datasets/lex_rel_valid.txt",
  "test_data_path": "extrinsic_tasks/datasets/lex_rel_test.txt",
  "evaluate_on_test": true,
  "model": {
    "type": "lex_rel_classifier",
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "pretrained_file": "${emb_filetxt}",
        "embedding_dim": ${task_emb_size},
        "trainable": false
      }
    },
    "shared_feedforward": {
      "input_dim": 300,
      "num_layers": 1,
      "hidden_dims": 1500,
      "activations": "relu",
      "dropout": 0.2
    },
    "classifier_feedforward_word1": {
      "input_dim": 1500,
      "num_layers": 1,
      "hidden_dims": 500,
      "activations": "linear",
      "dropout": 0.2
    },
    "classifier_feedforward_word2": {
      "input_dim": 1500,
      "num_layers": 1,
      "hidden_dims": 500,
      "activations": "linear",
      "dropout": 0.2
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["word1", "num_tokens"]],
    "batch_size": 1024
  },
  "trainer": {
    "num_epochs": 30,
    "patience": 3,
    "cuda_device": 0,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
