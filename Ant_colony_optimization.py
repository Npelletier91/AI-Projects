import numpy as np

distance_matrix = np.array([
    [4, 2, 4, 3, 5],
    [6, 1, 6, 7, 6],
    [4, 6, 4, 1, 3],
    [3, 7, 1, 4, 2],
    [5, 1, 2, 2, 8]
])

num_cities = distance_matrix.shape[0]
num_ants = 50


def ant_colony_optimization(num_iterations):
    pheromone_level = np.ones((num_cities, num_cities))
    heuristic_info = 1 / (distance_matrix + np.finfo(float).eps)
    alpha = 1.0
    beta = 2.0
    best_distance = float('inf')
    best_path = []
    best_iteration = -1

    for iteration in range(num_iterations):
        ant_paths = np.zeros((num_ants, num_cities), dtype=int)
        ant_distances = np.zeros(num_ants)

        for ant in range(num_ants):
            current_city = np.random.randint(num_cities)
            visited = [current_city]

            for _ in range(num_cities - 1):
                selection_probs = (pheromone_level[current_city] ** alpha) * (heuristic_info[current_city] ** beta)
                selection_probs[np.array(visited)] = 0

                next_city = np.random.choice(np.arange(num_cities), p=(selection_probs / np.sum(selection_probs)))
                ant_paths[ant, _ + 1] = next_city
                visited.append(next_city)

                ant_distances[ant] += distance_matrix[current_city, next_city]
                current_city = next_city

            ant_distances[ant] += distance_matrix[current_city, ant_paths[ant, 0]]

        pheromone_level *= 0.5  # Evaporation
        for ant in range(num_ants):
            for city in range(num_cities - 1):
                pheromone_level[ant_paths[ant, city], ant_paths[ant, city + 1]] += 1 / ant_distances[ant]

            pheromone_level[ant_paths[ant, -1], ant_paths[ant, 0]] += 1 / ant_distances[ant]

        min_distance_idx = np.argmin(ant_distances)
        if ant_distances[min_distance_idx] < best_distance:
            best_distance = ant_distances[min_distance_idx]
            best_path = ant_paths[min_distance_idx]
            best_iteration = iteration

    return best_path, best_distance, best_iteration

num_iterations = 100
best_path, best_distance, best_iteration = ant_colony_optimization(num_iterations)


print('Here is the best path:', best_path)
print('Here is the best distance:', best_distance)
print('Here is the best iteration:', best_iteration)