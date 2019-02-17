{
  "train_data_path": "", // need to add training data path here! i.e. nlvr2/data/train.json
  "validation_data_path": "",

  "dataset_reader": {
    "type": "nlvr_reader"
  },

  "model": {
    "type": "sentiment_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "",
          "embedding_dim": 300,
          "trainable": false
        }
      }
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 300,
      "hidden_size": 128,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 256,
      "num_layers": 2,
      "hidden_dims": [256, 2],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },

  "iterator": {
    "type": "basic"
  },

  "trainer": {
    "num_epochs": 15,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
