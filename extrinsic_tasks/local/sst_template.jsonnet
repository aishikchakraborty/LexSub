{
  "dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": true,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "validation_dataset_reader":{
    "type": "sst_tokens",
    "use_subtrees": false,
    "granularity": "2-class",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "train_data_path": "extrinsic_tasks/datasets/sst/train.txt",
  "validation_data_path": "extrinsic_tasks/datasets/sst/dev.txt",
  "test_data_path": "extrinsic_tasks/datasets/sst/test.txt",
  "evaluate_on_test": true,
  "model": {
    "type": "bcn",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "${emb_filetxt}",
            "type": "embedding",
            "embedding_dim": ${task_emb_size},
            "trainable": false
        }
      }
    },
    "embedding_dropout": 0.25,
    "pre_encode_feedforward": {
        "input_dim": ${task_emb_size},
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 2],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 100
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+accuracy",
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    }
  }
}
