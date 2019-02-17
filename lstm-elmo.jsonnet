{
  "train_data_path": "", // need to add training data path here! i.e. nlvr2/data/train.json
  "validation_data_path": "",

  "dataset_reader": {
    "type": "nlvr_reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },

  "model": {
    "type": "nlvr_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "",
          "embedding_dim": 100,
          "trainable": false
        },
        "elmo": {
          "type": "elmo_token_embedder",
          "options_file": "",
          "weight_file": "",
          "do_layer_norm": false,
          "dropout": 0.5
         }
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "input_size": 1124,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 100,
      "num_layers": 2,
      "hidden_dims": 100,
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },

  "iterator": {
    "type": "basic"
  },

  "trainer": {
    "num_epochs": 3,
    "patience": 10,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
