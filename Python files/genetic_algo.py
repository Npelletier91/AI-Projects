import random

TARGET_PHRASE =  "Hello, World!"
POPULATION_SIZE = 150
MUTATION_RATE = .05


def gerenate_population():
    population = []
    for _ in range(POPULATION_SIZE):
        individual = ''.join(random.choice('qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM ,.!') for _ in range(len(TARGET_PHRASE)))
        population.append(individual)
    return population

def calculate_fitness(individual):
    score = 0
    for i in range(len(TARGET_PHRASE)):
        if individual[i] == TARGET_PHRASE[i]:
            score += 1
    return score

def select_parents(population):
    parents = []
    for _ in range(2):
        parents.append(max(population, key=calculate_fitness))
    return parents

def crossover(parents):
    offspring = ""
    crossover_point = random.randint(0,len(TARGET_PHRASE) - 1)
    for i in range(len(TARGET_PHRASE)):
        if i <= crossover_point:
            offspring += parents[0][i]
        else:
            offspring += parents[1][i]
    return offspring

def mutate(offspring):
    mutated_offspring = ""
    for i in range(len(offspring)):
        if random.random() < MUTATION_RATE:
            mutated_offspring += random.choice('qwertyuiopasdfghjklzxcvbnm,. !QWERTYUIOPASDFGHJKLZXCVBNM')
        else:
            mutated_offspring += offspring[i]
    return mutated_offspring

def genetic_algorithm():
    population = gerenate_population()
    generation = 1

    while True:
        print(f"Generation {generation} - Best Fit: {max(population, key=calculate_fitness)}")

        if TARGET_PHRASE in population:
            break

        new_population = []
        for _ in range (POPULATION_SIZE // 2):
            parents = select_parents(population)
            offspring = crossover(parents)
            mutated_offspring = mutate(offspring)
            new_population.extend([offspring, mutated_offspring])

        population = new_population
        generation += 1


genetic_algorithm()