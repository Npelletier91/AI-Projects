my_array = [4,7,2,5,9,12]

def sorting_func():
    swap = True
    while (swap == True):
        swap = False
        for i in range(len(my_array) - 1):
            
            if (my_array[i] > my_array[i + 1]):
                swap_num = my_array[i]
                my_array[i] = my_array[i + 1]
                my_array[i + 1] = swap_num
                swap = True
    return my_array
        
sorted_array = sorting_func()           
print(sorted_array)