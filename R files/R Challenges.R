#1 Challenge 1

x = 10
x



######2 Challenge 2######
nums = 2:100
i = 1
result = 0

while (i < 100) {
  result = result + nums[i]
  i = i + 2
}
print(result)

# ALTERNATE ANSWER
sum_result = 0

for (i in seq(2,100, by=2)){
  sum_result = sum_result + i
}
print(sum_result)



######3 Challenge 3#######
my_func = function(n) {
  if (n == 0 || n==1) {
    return(1)
  } else {
      return(n * my_func(n - 1))
  }
}

n = 5
result_func = my_func(n)
cat("The result of", n, "factorial is:", result_func)




######4 Challenge 4########
count = 0
Vowel_Func = function(str){
  str = strsplit(str,"")
  for (letter in str) {
    if (letter == "a" || "e" || "i" || "o" || "u" || "y") {  #doesnt work for R
      count = count + 1
    }
  }
}
str = "Hello, World!"
Vowel_Func(str)

#ALTERNATE METHOD#
vowels = c("a",'e','i','o','u')
for (char in strsplit(str, "")[[1]]){
  if (char %in% vowels){
    count = count + 1
  }
}
print(count)





######5 Challenge 5###########
numbers = c(1,2,3,5,12,20)
weights = c(.2,.4,.6,1,2.4,4)

weight_sum = sum(numbers * weights)
total_sum = sum(weights)
average_sum = weight_sum / total_sum
cat("average weighed sum:", average_sum)

##alternate method
weighted_avg = weighted.mean(numbers,weights)




######## 6 CHALLENGE 6 ########
summary(iris)

plot(iris$Sepal.Length, iris$Sepal.Width, xlab = "Length", ylab = "Width", main = "Scatter plot of length vs width")





###### 7 Challenge 7 #########
# LINEAR REGRESSION MODEL - PREDICTION

data(mtcars)

car_model = lm(mpg ~ hp, data = mtcars)
plot(car_model)

new_car = data.frame(hp = 150)
new_car_prediction = predict(car_model, newdata = new_car)


model3 = lm(mpg ~ disp, data = mtcars)
new_car3 = data.frame(disp = 300)
preditcion3 = predict(model3, newdata = new_car3)
preditcion3


###### 8 CHALLENGE 8 ###########
# K NEAREST NEIGHBORS
data(iris)
set.seed(123)
library(caret)
library(ggplot2)
library(lattice)
sample_indices = sample(1:nrow(iris), 0.7 * nrow(iris))

training_data = iris[sample_indices, ]
testing_data = iris[-sample_indices, ]

library(class)

model = knn(train = training_data[,-5], test = testing_data[,-5], cl = training_data[,5], k = 3)
accuracy = sum(model == testing_data[,5]) / nrow(testing_data)






###### 9 Challenge #####
# PRINCIPLE COMPONENT ANALYSIS   - PCA
PCA_result = prcomp(mtcars[, -c(2,4,6,8,10)], scale = TRUE)
PC1 = PCA_result$x[,1]
PC2 = PCA_result$x[,2]

plot(PC1, PC2)
PC1





###### 10 Challenge 10 #########
# NAIVE BAYES CLASSIFIER
# Accuracy test
library(e1071)
library(tm)

seed = 123
set.seed(seed)

train_data = sample(1:nrow(iris), 0.7 * nrow(iris))


training_data = iris[train_data, ]
testing_data = iris[-train_data, ]


naive_model = naiveBayes(Species ~ ., data = training_data)
predicted = predict(naive_model, newdata = testing_data)

accuracy = sum(predicted == testing_data$Species) / nrow(testing_data) * 100
cat("The accuray of the seed", seed, "model is:", accuracy, "%")


# Prediction test
trainer_data = iris[ ,-5]
nb_model = naiveBayes(Species ~ ., trainer_data)

new_plant = data.frame(
  Sepal.Length = 3.5,
  Sepal.Width = 5,
  Petal.Length = 4,
  Petal.Width = 6
)

new_prediction = predict(nb_model, new_plant)
cat("The new data species is", levels(iris$Species)[new_prediction])

    















