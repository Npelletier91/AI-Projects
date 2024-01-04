data(mtcars)
cars = mtcars
subset_data = cars[c("mpg", "cyl", "hp")]

set.seed(42)

train_indices <- sample(
  x = nrow(subset_data),
  size = nrow(subset_data) * .7
)

train_data <- subset_data[train_indices, ]
test_data <- subset_data[-train_indices, ]


model <- lm(
  mpg ~ ., data = train_data
)

summary(model)

new_car <- data.frame(mpg = NA, cyl = 6, hp = 150)
predicted_mpg <- predict(model, newdata = new_car)

cat("New car mpg is: ", predicted_mpg)




print(cars)
print(cars[ , 1])
