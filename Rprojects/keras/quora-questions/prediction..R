library(keras)
library(dplyr)
library(tfestimators)


model <- load_model_hdf5("model-question-pairs-with-flgas.hdf5", compile = FALSE)
tokenizer <- load_text_tokenizer("tokenizer-question-pairs")

predict_question_pairs <- function(model, tokenizer, q1, q2) {
  q1 <- texts_to_sequences(tokenizer, list(q1))
  q2 <- texts_to_sequences(tokenizer, list(q2))
  
  q1 <- pad_sequences(q1, 20)
  q2 <- pad_sequences(q2, 20)
  
  as.numeric(predict(model, list(q1, q2)))
}



predict_question_pairs(
  model,
  tokenizer,
  "What's R programming?",
  "What's R in programming?"
)