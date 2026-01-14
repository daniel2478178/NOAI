# -*- coding: utf-8 -*-
"""
遗传算法解决旅行商问题(TSP)
选择策略：赌轮盘选择(启用) / 锦标赛选择(注释备用)
交叉策略：部分映射交叉PMX(启用) / 循环交叉CX(注释备用)
变异策略：移位变异(启用) / 互换变异(注释备用)
author: Gao Chaoyu  |  适配需求定制版
"""
import numpy as np
import matplotlib.pyplot as plt

# ====================== 1. 超参数设置（适配7个城市） ======================
np.random.seed(666)       # 随机种子，保证结果可复现
city_num = 7              # 城市数量改为7
pop_size = 50             # 种群规模适当减小
iter_max = 100            # 迭代次数适当减少
cross_prob = 0.85         # 交叉概率
mutate_prob = 0.1         # 变异概率
elite_num = 2             # 精英保留数

# ====================== 2. 7个城市的固定坐标 & 计算距离矩阵 ======================
# 7个城市的坐标（模拟真实城市分布，如：北京、上海、广州等，用编号0-6表示）
city_coords = np.array([
    [10, 20],   # 城市0
    [30, 50],   # 城市1
    [50, 30],   # 城市2
    [70, 60],   # 城市3
    [90, 40],   # 城市4
    [20, 70],   # 城市5
    [80, 20]    # 城市6
])
# 计算距离矩阵：两两城市之间的欧式距离
distance_mat = np.zeros((7, 7))
for i in range(7):
    for j in range(7):
        distance_mat[i][j] = np.sqrt(np.sum((city_coords[i] - city_coords[j]) ** 2))


# ====================== 3. 种群初始化：生成合法的路径（无重复城市） ======================
def init_population(pop_size, city_num):
    population = []
    for _ in range(pop_size):
        # 生成一条[0,1,..,city_num-1]的随机排列 = 一条合法的旅行路径
        path = np.random.permutation(city_num)
        population.append(path)
    return np.array(population)


# ====================== 4. 适应度函数：TSP的适应度 = 路径总距离的倒数 ======================
# 注：TSP是求总距离最小值，遗传算法是保留适应度高的个体，所以用倒数，路径越短→适应度越高
def calc_fitness(population, distance_mat):
    fitness = []
    for path in population:
        total_dist = 0
        # 计算路径总距离：依次累加相邻城市距离 + 回到起点的距离
        for i in range(len(path) - 1):
            total_dist += distance_mat[path[i]][path[i + 1]]
        total_dist += distance_mat[path[-1]][path[0]]
        fitness.append(1 / total_dist)  # 适应度 = 1/总距离
    return np.array(fitness)


# ====================== 5. 选择策略 二选一 ======================
# ----- 5.1 赌轮盘选择（默认启用，重点） -----
def roulette_selection(population, fitness):
    # 计算每个个体的选择概率 = 个体适应度 / 种群总适应度
    fit_sum = np.sum(fitness)
    select_prob = fitness / fit_sum
    # 计算累积概率
    cum_prob = np.cumsum(select_prob)
    pop_size = len(population)
    new_pop = []
    # 轮盘赌选择pop_size次，生成新种群
    for _ in range(pop_size):
        r = np.random.random()  # 生成0~1的随机数
        # 找到随机数落在的累积概率区间，选择对应个体
        for i in range(pop_size):
            if r <= cum_prob[i]:
                new_pop.append(population[i].copy())
                break
    return np.array(new_pop)


# ----- 5.2 锦标赛选择（注释备用，取消注释即可替换使用） -----
# def tournament_selection(population, fitness, tournament_size=3):
#     pop_size = len(population)
#     new_pop = []
#     for _ in range(pop_size):
#         # 随机选择k个个体作为锦标赛选手
#         idx = np.random.choice(pop_size, tournament_size, replace=False)
#         tournament_pop = population[idx]
#         tournament_fit = fitness[idx]
#         # 选择锦标赛中适应度最高的个体
#         best_idx = np.argmax(tournament_fit)
#         new_pop.append(tournament_pop[best_idx].copy())
#     return np.array(new_pop)

# ====================== 6. 交叉策略 二选一 (排列编码专用，无重复城市) ======================
# ----- 6.1 部分映射交叉 PMX（默认启用，重点，TSP最常用） -----
def PMX_crossover(parent1, parent2):
    city_num = len(parent1)
    # 随机选择两个交叉点
    cross_point1 = np.random.randint(0, city_num - 1)
    cross_point2 = np.random.randint(cross_point1 + 1, city_num)

    child1 = parent1.copy()
    child2 = parent2.copy()

    # 建立映射关系
    map1 = {}
    map2 = {}
    for i in range(cross_point1, cross_point2):
        map1[parent2[i]] = parent1[i]
        map2[parent1[i]] = parent2[i]

    # 处理交叉段外的基因，避免重复
    for i in range(city_num):
        if cross_point1 <= i < cross_point2:
            continue
        # 处理子代1
        while child1[i] in map1:
            child1[i] = map1[child1[i]]
        # 处理子代2
        while child2[i] in map2:
            child2[i] = map2[child2[i]]

    # 交换交叉段
    child1[cross_point1:cross_point2] = parent2[cross_point1:cross_point2]
    child2[cross_point1:cross_point2] = parent1[cross_point1:cross_point2]

    return child1, child2


# ----- 6.2 循环交叉 CX（注释备用，取消注释即可替换使用） -----
# def CX_crossover(parent1, parent2):
#     city_num = len(parent1)
#     child1 = np.full(city_num, -1)
#     child2 = np.full(city_num, -1)
#     idx = 0
#     # 构建第一个循环
#     while child1[idx] == -1:
#         child1[idx] = parent1[idx]
#         child2[idx] = parent2[idx]
#         idx = np.where(parent1 == parent2[idx])[0][0]
#     # 填充剩余位置
#     for i in range(city_num):
#         if child1[i] == -1:
#             child1[i] = parent2[i]
#             child2[i] = parent1[i]
#     return child1, child2

# 种群批量交叉操作
def crossover_population(population, cross_prob):
    pop_size = len(population)
    new_pop = []
    i = 0
    while i < pop_size:
        # 随机配对，判断是否交叉
        if np.random.random() <= cross_prob:
            child1, child2 = PMX_crossover(population[i], population[i + 1])
            new_pop.append(child1)
            new_pop.append(child2)
        else:
            new_pop.append(population[i].copy())
            new_pop.append(population[i + 1].copy())
        i += 2
    return np.array(new_pop)


# ====================== 7. 变异策略 二选一 (排列编码专用，无重复城市) ======================
# ----- 7.1 移位变异（默认启用，重点） -----
def shift_mutation(individual):
    city_num = len(individual)
    # 随机选择一个起始位置和移位长度
    start = np.random.randint(0, city_num - 2)
    length = np.random.randint(2, city_num - start)
    # 选中片段整体移位一位
    individual[start:start + length] = np.roll(individual[start:start + length], 1)
    return individual


# ----- 7.2 互换变异（注释备用，取消注释即可替换使用） -----
# def swap_mutation(individual):
#     city_num = len(individual)
#     # 随机选择两个不同的位置
#     pos1 = np.random.randint(0, city_num)
#     pos2 = np.random.randint(0, city_num)
#     while pos1 == pos2:
#         pos2 = np.random.randint(0, city_num)
#     # 交换两个位置的城市
#     individual[pos1], individual[pos2] = individual[pos2], individual[pos1]
#     return individual

# 种群批量变异操作
def mutate_population(population, mutate_prob):
    new_pop = []
    for ind in population:
        if np.random.random() <= mutate_prob:
            new_ind = shift_mutation(ind.copy())
            new_pop.append(new_ind)
        else:
            new_pop.append(ind.copy())
    return np.array(new_pop)


# ====================== 8. 精英保留策略 ======================
# 保留种群中适应度最高的elite_num个个体，直接进入下一代，保证最优解不丢失
def elite_reserve(population, fitness, elite_num):
    elite_idx = np.argsort(fitness)[-elite_num:]  # 适应度最高的个体索引
    elite_individuals = population[elite_idx].copy()
    return elite_individuals


# ====================== 9. 计算路径总距离（适配TSP结果输出） ======================
def calc_path_dist(individual, distance_mat):
    total_dist = 0
    city_num = len(individual)
    for i in range(city_num - 1):
        total_dist += distance_mat[individual[i]][individual[i + 1]]
    total_dist += distance_mat[individual[-1]][individual[0]]
    return total_dist


# ====================== 10. 主函数：遗传算法执行流程 ======================
def GA_TSP():
    # 初始化种群
    population = init_population(pop_size, city_num)
    # 记录每代最优距离，用于绘制收敛曲线
    best_dist_list = []

    for iter_idx in range(iter_max):
        # 1. 计算适应度
        fitness = calc_fitness(population, distance_mat)
        # 2. 记录当前代最优解
        best_idx = np.argmax(fitness)
        best_ind = population[best_idx]
        best_dist = calc_path_dist(best_ind, distance_mat)
        best_dist_list.append(best_dist)

        # 打印迭代信息
        if (iter_idx + 1) % 20 == 0:
            print(f"迭代次数: {iter_idx + 1:3d} | 最优路径总距离: {best_dist:.2f}")

        # 3. 精英保留
        elite_ind = elite_reserve(population, fitness, elite_num)

        # 4. 选择操作：默认赌轮盘，替换成 tournament_selection() 即可切换锦标赛
        population = roulette_selection(population, fitness)

        # 5. 交叉操作
        population = crossover_population(population, cross_prob)

        # 6. 变异操作
        population = mutate_population(population, mutate_prob)

        # 7. 替换种群最后elite_num个个体为精英，保证精英存活
        population[-elite_num:] = elite_ind

    # 迭代结束，获取最终最优解
    final_fitness = calc_fitness(population, distance_mat)
    final_best_idx = np.argmax(final_fitness)
    final_best_ind = population[final_best_idx]
    final_best_dist = calc_path_dist(final_best_ind, distance_mat)

    print("=" * 50)
    print(f"遗传算法迭代完成！")
    print(f"最优旅行路径: {final_best_ind}")
    print(f"最优路径总距离: {final_best_dist:.2f}")
    return final_best_ind, best_dist_list


# ====================== 11. 结果可视化 ======================
def plot_result(best_path, best_dist_list):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False

    # 子图1：迭代收敛曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(iter_max), best_dist_list, 'b-', linewidth=1.5)
    plt.title('遗传算法迭代收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('最优路径总距离')
    plt.grid(alpha=0.3)

    # 子图2：最优路径可视化
    plt.subplot(1, 2, 2)
    # 绘制城市坐标点
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c='r', s=60, label='城市')
    # 绘制最优路径
    for i in range(city_num - 1):
        x1, y1 = city_coords[best_path[i]]
        x2, y2 = city_coords[best_path[i + 1]]
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.8)
    # 绘制回到起点的路径
    x1, y1 = city_coords[best_path[-1]]
    x2, y2 = city_coords[best_path[0]]
    plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.8)
    # 标注城市编号
    for i in range(city_num):
        plt.text(city_coords[i, 0] + 1, city_coords[i, 1] + 1, str(i), fontsize=10)
    plt.title(f'TSP最优路径 (城市数={city_num})')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ====================== 运行主程序 ======================
if __name__ == '__main__':
    best_path, dist_list = GA_TSP()
    plot_result(best_path, dist_list)