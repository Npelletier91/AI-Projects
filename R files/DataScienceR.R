l <- TRUE
i <- 123L
n <- 1.23
c <- "ABC 123"
d <- as.Date("2024-01-03")
d

f <- function(x){
  x + 1
}
f(2)

vector <- c(1,2,3)
vector

s <- 1:5
s

matrix <- matrix(
  data = 1:6, nrow = 2, ncol = 3
)
matrix

l <- list(TRUE, 123L, 2.34, "abc")
l

df <- data.frame(
  Name = c("Cat", "Dog"),
  HowMany = c(5,10),
  IsPet = c(TRUE, TRUE)
)
df
df[1,2]
df$HowMany

library(dplyr)
