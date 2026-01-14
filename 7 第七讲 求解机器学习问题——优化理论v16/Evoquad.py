import numpy as np


def gene(x):
    return list(map(int, format(int(x), "05b")))

def degene(x):
    return int("".join(map(str, x)), 2) if type(x) != np.int64 else x

def fitness_calc(individual):
    return degene(individual) ** 2

def cross_breed(parent1, parent2):
 
    a = np.random.randint(0, len(parent1))
    b = np.random.randint(a,len(parent1)) 
    parent1[a:b], parent2[a:b] = parent2[a:b], parent1[a:b] 
    return parent1, parent2

def roulette_selection(population):
    fitness = np.apply_along_axis(lambda r: fitness_calc(r), 1, population)
    select_prob = fitness / np.sum(fitness)
    cum_prob = np.cumsum(select_prob)
    new_pop = []
    # 轮盘赌选择pop_size次，生成新种群
    for _ in range(len(population)):
        r = np.random.random()  # 生成0~1的随机数
        # 找到随机数落在的累积概率区间，选择对应个体
        for i in range(len(population)):
            if r <= cum_prob[i]:
                new_pop.append(population[i].copy())
                break
    return np.array(new_pop)
def tournament_selection(population,k=3, reqiured_num = 3):
    current_parent_num = 0
    newpop = np.array(population)
    #print(newpop.shape,fitness.shape)
    while current_parent_num < reqiured_num:
        population = newpop.copy()
        fitness = np.array([fitness_calc(ind) for ind in population])   

        wherechose = np.random.choice(range(len(newpop)), size=k, replace=False)
       # print(newpop,wherechose)
        tournament = newpop[wherechose]


        Survivorind, survivorfitness = tournament[np.argmax(fitness[wherechose])], fitness[wherechose][np.argmax(fitness[wherechose])]
        #newpop = np.delete(newpop,wherechose,axis = 0)        
        current_parent_num +=1
        #fitness = np.delete(fitness,wherechose)
        newpop = np.concatenate([newpop, Survivorind.reshape(1,-1)], axis=0)
    return np.array(newpop)

def mutate(population):
    for i in population:
        if np.random.random() < 0.2:
            i[np.random.randint(0,len(population[0]))] ^= 1
    fitness = np.array([fitness_calc(ind) for ind in population])
    return population


def initpopulation(pop_size):
    arr = np.random.randint(0, 32, pop_size)
    populatio = [gene(x) for x in arr]
    return populatio

population = initpopulation(10)
for round in range(10):
    #population = roulette_selection(population)

    population = tournament_selection(population)
    p1 = population[np.random.randint(0,len(population))]
    p2 = population[np.random.randint(0,len(population))]
    p1,p2 = cross_breed(p1, p2)
    population = np.concatenate([population, [p1, p2]], axis=0)
    population = mutate(population)

    
print("最优解为：",np.max(np.array([fitness_calc(ind) for ind in population])   ))