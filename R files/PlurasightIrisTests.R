data(iris)

plot(
  x = iris$Petal.Length,
  y = iris$Petal.Width,
  main = "legth vs width",
  xlab = "petal length",
  ylab = "petal width"
)

#corelation between length and width of petal
cor(
  x = iris$Petal.Length,
  y = iris$Petal.Width
)

library(ff)
install.packages("biglm")
install.packages("ff")
library(biglm)


irisff <- read.table.ffdf(
  file = "Iris.csv",
  FUN = "read.csv")





model <- biglm(
  formula = Petal.Width ~ Petal.Length,
  data = irisff
)

summary(model)

plot(
  x = irisff$Petal.Length[],
  y = irisff$Petal.Width[]
)

b <- summary(model)$mat[1,1]

m <- summary(model)$mat[2,1]

lines(
  x = irisff$Petal.Length[],
  y = m * irisff$Petal.Length[] + b,
  col = "red",
  lwd = 3
)

predic(
  object = model,
  newdata = data.frame(
    Petal.Length = c(2,5,7),
    Petal.Width = c(0,0,0)
  )
)

