library(tfruns)

runs <- tuning_run(
  "keras/duplicated-pairs-quora.R", 
  flags = list(
    vocab_size = c(40000, 50000, 60000),
    max_len_padding = c(15, 20),
    embedding_size = c(128, 256),
    regularization = c(0.0001, 0.001),
    seq_embedding_size = c(128, 256, 512)
  ), 
  runs_dir = "tuning", 
  sample = 0.2
)
cloudml_train("keras/quora-questions/duplicated-pairs-quora.R") 