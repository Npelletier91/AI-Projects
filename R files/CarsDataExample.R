cars <- mtcars

setwd("C:/Users/Nicol/Documents/GitHub Repos/AI-Projects/R files")

head(cars)

library(dplyr)

temp <- select(
  .data = cars,
  am,
  cyl,
  mpg
)

head(temp)

temp <- filter(
  .data = temp,
  am == 0
)

temp <- mutate(
  .data = temp,
  Consumption = mpg * 0.425
)

head(temp)

temp <- group_by(
  .data =temp,
  cyl
)

head(temp)

temp <- summarize(
  .data = temp,
  Avg.Consumption = mean(Consumption)
)

head(temp)


temp <- arrange(
  .data = temp,
  desc(Avg.Consumption)
)

head(temp)

effeciency <- as.data.frame(temp)

print(effeciency)


write.csv(
  x = effeciency,
  file = "Fuel Efficiency File",
  row.names = FALSE
)


cars <- mtcars
library(ggplot2)

ggplot(
  data = cars,
  aes(x = am)) +
  geom_bar() +
  ggtitle("Count of cars by transmission type") +
  xlab("Transmission Type") +
  ylab("Count of Cars")

