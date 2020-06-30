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

questions_length <- c(
  map_int(question1, length),
  map_int(question2, length)
)

quantile(questions_length, c(0.60, 0.75, 0.8, 0.9, 0.95, 0.99))

question1_padded <- pad_sequences(question1, maxlen = FLAGS$max_len_padding, value = 50000 + 1)
question2_padded <- pad_sequences(question2, maxlen = FLAGS$max_len_padding, value = 50000 + 1)


perc_words_question1 <- map2_dbl(question1, question2, ~mean(.x %in% .y))
perc_words_question2 <- map2_dbl(question2, question1, ~mean(.x %in% .y))


df_model <- data.frame(
  perc_words_question1 = perc_words_question1,
  perc_words_question2 = perc_words_question2,
  is_duplicate = quora_df$is_duplicate
) %>%
  na.omit()

val_sample <- sample.int(nrow(df_model), 0.1*nrow(df_model))
logistic_regression <- glm(
  is_duplicate ~ perc_words_question1 + perc_words_question2, 
  family = "binomial",
  data = df_model[-val_sample,]
)
summary(logistic_regression)