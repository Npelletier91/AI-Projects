my_array = c(4,7,2,5,9,12)

sorting_func = function(my_array) {
  n = length(my_array)
  
  for (i in 1:(n - 1)) {
    for (j in 1:(n - i)) {
      if (my_array[j] > my_array[j + 1]) {
        temporary = my_array[j]
        my_array[j] = my_array[j + 1]
        my_array[j + 1] = temporary
      }
    }
  }
  print(my_array)
}

sorting_func(my_array)