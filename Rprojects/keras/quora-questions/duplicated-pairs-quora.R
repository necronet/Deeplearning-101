# Classifying Duplicate Questions from Quora with Keras

library(keras)
library(readr)
library(purrr)

FLAGS <- flags(
  flag_integer("vocab_size", 50000),
  flag_integer("max_len_padding", 20),
  flag_integer("embedding_size", 256),
  flag_numeric("regularization", 1e-4),
  flag_integer("seq_embedding_size", 512)
)

quora_data <- get_file("quora_duplicate_questions.tsv","http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv")

quora_df <- read_tsv(quora_data)

tokenizer <- text_tokenizer(num_words = FLAGS$vocab_size)
tokenizer %>% fit_text_tokenizer(unique(c(quora_df$question1, quora_df$question2)))

question1 <- texts_to_sequences(tokenizer, quora_df$question1)
question2 <- texts_to_sequences(tokenizer, quora_df$question2)

question1_padded <- pad_sequences(question1, maxlen = 20, value = FLAGS$vocab_size + 1)
question2_padded <- pad_sequences(question2, maxlen = 20, value = FLAGS$vocab_size + 1)

input1 <- layer_input(shape = c(FLAGS$max_len_padding))
input2 <- layer_input(shape = c(FLAGS$max_len_padding))

word_embedder <- layer_embedding(
  input_dim = FLAGS$vocab_size + 2, 
  output_dim = FLAGS$embedding_size, 
  input_length = FLAGS$max_len_padding, 
  embeddings_regularizer = regularizer_l2(l = FLAGS$regularization)
)

seq_embedder <- layer_lstm(
  units = FLAGS$seq_embedding_size, 
  recurrent_regularizer = regularizer_l2(l = FLAGS$regularization)
)

vector1 <- input1 %>% word_embedder() %>% seq_embedder()
vector2 <- input2 %>% word_embedder() %>% seq_embedder()

cosine_similarity <- layer_dot(list(vector1, vector2), axes = 1)

output <- cosine_similarity %>% layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(list(input1, input2), output)
model %>% compile(
  optimizer = "adam", 
  metrics = list(acc = metric_binary_accuracy), 
  loss = "binary_crossentropy"
)

set.seed(1817328)
val_sample <- sample.int(nrow(question1_padded), size = 0.1*nrow(question1_padded))

train_question1_padded <- question1_padded[-val_sample,]
train_question2_padded <- question2_padded[-val_sample,]
train_is_duplicate <- quora_df$is_duplicate[-val_sample]

val_question1_padded <- question1_padded[val_sample,]
val_question2_padded <- question2_padded[val_sample,]
val_is_duplicate <- quora_df$is_duplicate[val_sample]

model %>% fit(
  list(train_question1_padded, train_question2_padded),
  train_is_duplicate, 
  batch_size = 128, 
  epochs = 2, 
  validation_data = list(
    list(val_question1_padded, val_question2_padded), 
    val_is_duplicate
  ),
  callbacks = list(
    callback_early_stopping(patience = 5),
    callback_reduce_lr_on_plateau(patience = 3)
  )
)

save_model_hdf5(model, "model-question-pairs-with-flgas.hdf5")
save_text_tokenizer(tokenizer, "tokenizer-question-pairs")