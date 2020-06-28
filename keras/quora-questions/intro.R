# Based on: https://blogs.rstudio.com/ai/posts/2018-01-09-keras-duplicate-questions-quora/

library(keras)

mnist <- dataset_mnist()

l = 36
randomIndex <- sample(1:length(mnist$train$y), l)
par(mfcol = c(6, 6))
par(mar=c(0, 0, 3, 0), xaxs='i',yaxs='i')
for (i in 1:l) {
  im <- t(apply(mnist$train$x[randomIndex[i], , ], 2, rev))
  image(im, main = paste(mnist$train$y[randomIndex[i]]), xaxt='n', col=gray((0:255)/255))
}

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'sigmoid', input_shape = c(784),  kernel_initializer='random_normal')  %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'sigmoid') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

model %>% evaluate(x_test, y_test)

model %>% predict_classes(x_test)










