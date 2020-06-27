
quora_data <- get_file("quora_duplicate_questions.tsv","http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv")

quora_df <- read_tsv(quora_data)

tokenizer <- text_tokenizer(num_words = 50000)
tokenizer %>% fit_text_tokenizer(unique(c(quora_df$question1, quora_df$question2)))


question1 <- texts_to_sequences(tokenizer, quora_df$question1)
question2 <- texts_to_sequences(tokenizer, quora_df$question2)

questions_length <- c(
  map_int(question1, length),
  map_int(question2, length)
)

quantile(questions_length, c(0.60, 0.75, 0.8, 0.9, 0.95, 0.99))

question1_padded <- pad_sequences(question1, maxlen = 20, value = 50000 + 1)
question2_padded <- pad_sequences(question2, maxlen = 20, value = 50000 + 1)


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