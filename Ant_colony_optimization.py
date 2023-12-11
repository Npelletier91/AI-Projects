import numpy as np

distance_matrix = np.array([
    [0,2,4,3,5],
    [2,0,6,7,8],
    [4,6,0,1,3],
    [3,7,1,0,1],
    [5,8,3,2,0]
])

num_cities = distance_matrix.shape[0]

num_ants = 10


def ant_colony_optimization(num_iterations):
    
    pheromone_level = np.ones((num_cities, num_cities))

    heuristic_info = 1 / (distance_matrix + np.finfo(float).eps) #avoid division by zero


    alpha = 1.0 #pheromone importance
    beta = 2.0 #heuristic importance

    best_distance = float('inf')
    best_path = []

    for iteration in range(num_iterations):
        ant_paths = np.zeros((num_ants,num_cities), dtype=int)
        ant_distances = np.zeros(num_ants)

        for ant in range(num_ants):
            #choose starting city
            current_city = np.random.randint(num_cities)
            visited = [current_city]
            
            #contruct path
            for _ in range(num_cities - 1):
                selection_probs = (pheromone_level[current_city] ** alpha) * (heuristic_info[current_city] ** beta)
                selection_probs[np.array(visited)] = 0  # Set selection probabilities of visited cities to 0

                next_city = np.random.choice(np.arange(num_cities), p=(selection_probs / np.sum(selection_probs)))

                # update the path visited list
                ant_paths[ant, _+1] = next_city
                visited.append(next_city)

                #update the distance
                current_city = next_city
            #update the distance to return to the starting city
            ant_distances[ant] += distance_matrix[current_city, ant_paths[ant, 0]]
        
        #update the pheromone level based on the ant paths
        pheromone_level *= 0.5 #evaporation
        for ant in range(num_ants):
            for city in range(num_cities - 1):
                pheromone_level[ant_paths[ant,city],ant_paths[ant,city+1]] += 1 / ant_distances[ant]
            pheromone_level[ant_paths[ant, -1], ant_paths[ant, 0]] += 1 / ant_distances[ant]

        # update the best path and discatnce if a btter solution is found
        mind_distance_idx = np.argmin(ant_distances)
        if ant_distances[mind_distance_idx] < best_distance:
            best_distance = ant_distances[mind_distance_idx]
            best_path = ant_paths[mind_distance_idx]

    return best_path, best_distance

num_iterations = 100
best_path, best_distance = ant_colony_optimization(num_iterations)

print("Here is the best path:", best_path)
print("Here is the ebst distance:", best_distance)

