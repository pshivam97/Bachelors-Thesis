#Importing required modules
import math
import random
import matplotlib.pyplot as plt
import copy

#First function to optimize
def function1(x):
    value = x**2
    return value

#Second function to optimize
def function2(x):
    value = (x-2)**2
    return value

#Function to find index of list
def index_of(a,List): #returns index of "a" in "list" , else "-1"
    for i in range(len(List)):
        if List[i] == a:
            return i
    return -1

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2, rank):
    S=[[] for i in range(len(values1))]
    front = [[]]
    n = [0 for i in range(len(values1))]

    for p in range(len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (values1[p] <= values1[q] and values2[p] < values2[q]) or (values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (values1[q] <= values1[p] and values2[q] < values2[p]) or (values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i] != [] :
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0 :
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to sort by values
def sort_by_values(front, values):
    sorted_list = []
    while len(sorted_list) != len(front) :
        sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(len(front))]
    front_values1 = []
    front_values2 = []

    for i in front :
        front_values1.append(values1[i])
        front_values2.append(values2[i])

    sorted1 = sort_by_values(front, front_values1[:])
    sorted2 = sort_by_values(front, front_values2[:])
    # sorted3 = sort_by_values(front, values3[:])

    distance1 = [0 for i in range(len(front))]
    distance2 = [0 for i in range(len(front))]
    distance1[sorted1[0]] = 4444444444444444  ### BIG NUMBER
    distance2[sorted2[0]] = 4444444444444444  ### BIG NUMBER


    for k in range(1,len(front)-1) :
        distance1[sorted1[k]] = (front_values1[sorted1[k+1]] - front_values1[sorted1[k-1]])/(max(front_values1)-min(front_values1) + 1)
        distance2[sorted2[k]] = (front_values2[sorted2[k+1]] - front_values2[sorted2[k-1]])/(max(front_values2)-min(front_values2) + 1)

    for k in range(len(front)):
        distance[k] = distance1[k] + distance2[k]

    # for k in range(1,len(front)-1) :
    #     distance[k] = distance[k] + (values3[sorted3[k+1]] - values3[sorted3[k-1]])/(max(values3) - min(values3))

    return distance

#Function to carry out the crossover using Simulated Binary Crossover
def crossover_and_mutation(p1,p2): # p1 is parent1 (real number) | p2 is parent2 (real number)
    weight = 3*(2*random.random() - 1)

    #### SBX Opeartor ####
    # beta = 0
    # n = 3
    # r = random.random()
    #
    # if r <= 0.5 :
    #     beta = (2*r)**(1/(n+1))
    # else :
    #     beta = (1/(2*(1-r)))**(1/(n+1))
    #
    # child1 = 0.5*((1+beta)*p1 + (1-beta)*p2) # child1
    # child2 = 0.5*((1-beta)*p1 + (1+beta)*p2) # child2

    child1 = (weight * p1) + ((1-weight) * p2)
    child2 = ((1-weight) * p1) + (weight * p2)

    (child1,child2) = mutation(child1,child2)
    return (child1,child2)

#Function to carry out the mutation operator
def mutation(c1,c2):
    mutation_prob_1 = random.random()
    mutation_prob_2 = random.random()

    if mutation_prob_1 < mutationRate :
        c1 += (2*random.random() - 1)*(c1/100)
    if mutation_prob_2 < mutationRate :
        c2 += (2*random.random() - 1)*(c2/100)

    return (c1,c2)

def binary_tournament_selection(fronts,previous_pool,current_pool,crowding_values,rank,crowding_flag) :

    if crowding_flag : # CROWDING FLAG = FALSE
        p1 = random.randint(0,pop_size-1)
        p2 = random.randint(0,pop_size-1)
        r1 = index_of(current_pool[p1],previous_pool)
        r2 = index_of(current_pool[p2],previous_pool)

        if rank[r1] < rank[r2] :
            return p1
        elif rank[r1] > rank[r2] :
            return p2
        else :
            crowding_distances = [crowding_values[rank[r1]][index_of(r1,fronts[rank[r1]])], crowding_values[rank[r2]][index_of(r2,fronts[rank[r2]])]]
            max_crowding_dist = max(crowding_distances)
            i = index_of(max_crowding_dist,crowding_distances)

            if i == 0:
                return p1
            else :
                return p2

    else :
        p1 = random.randint(0,pop_size-1)
        p2 = random.randint(0,pop_size-1)

        if rank[p1] < rank[p2] :
            return p1
        elif rank[p1] > rank[p2] :
            return p2
        else :
            r = random.random()
            if r <= 0.5 :
                return p1
            else :
                return p2

if __name__ == '__main__' :

    #######################    Main program starts here   #########################
    pop_size = 40
    max_gen = 100
    mutationRate = 0.2
    rank = [0 for i in range(pop_size)]

    #Initialization
    min_x = -1000
    max_x = 1000
    population = [min_x + (max_x-min_x)*random.random() for i in range(pop_size)]
    P_t = copy.deepcopy(population) ## Population list initialized of size = pop_size

    function1_values = [function1(population[i]) for i in range(pop_size)]
    function2_values = [function2(population[i]) for i in range(pop_size)]

    if max_gen == 0 :
        print("FINAL SOLUTION : \n",population,"\n\nMax Data Point: ",max(population)," \nMin Data Point: ",min(population))
        plt.xlabel('x^2', fontsize=15)
        plt.ylabel('(x-2)^2', fontsize=15)
        plt.scatter(function1_values, function2_values)
        plt.grid()
        plt.show()
        exit()

    fronts = fast_non_dominated_sort(function1_values[:], function2_values[:], rank)
    crowding_distance_values = []
    previous_R_t = None

    while len(population) != 2 * pop_size :
        parent1 = binary_tournament_selection(fronts,previous_R_t,P_t,crowding_distance_values,rank,False)
        parent2 = binary_tournament_selection(fronts,previous_R_t,P_t,crowding_distance_values,rank,False)

        (child1,child2) = crossover_and_mutation(P_t[parent1],P_t[parent2])
        population.append(child1)
        if len(population) == 2*pop_size :
            break
        else :
            population.append(child2)

    R_t = population

    generation_no = 0
    # no_of_generations = []
    # average_crowding_distance_objective_space = []
    # average_crowding_distance_data_space = []

    while generation_no < max_gen :

        # variable_space = copy.deepcopy(R_t)
        function1_values = [function1(R_t[i]) for i in range(2*pop_size)]
        function2_values = [function2(R_t[i]) for i in range(2*pop_size)]

        rank = [0 for i in range(2*pop_size)]

        fronts = fast_non_dominated_sort(function1_values[:],function2_values[:],rank)

        del crowding_distance_values[:]

        for i in range(len(fronts)):
            crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],fronts[i][:]))

        new_solution = []
        for i in range(len(fronts)) :
            if (pop_size - len(new_solution)) >= len(fronts[i]) : ## If the current front can be fully accomodated in new generation
                for each_value in fronts[i] :
                    new_solution.append(each_value)

            elif (len(new_solution) == pop_size):
                break

            else : ## If the current front cannot be fully accomodated in the new generation, then we decide by crowding distance to preserve diversity
                last_front_to_be_considered = [index_of(fronts[i][j],fronts[i]) for j in range(0,len(fronts[i]))]
                front_sorted = sort_by_values(last_front_to_be_considered, crowding_distance_values[i][:])
                front = [fronts[i][front_sorted[j]] for j in range(0,len(fronts[i]))]
                front.reverse()

                for each_value in front:
                    if len(new_solution) != pop_size :
                        new_solution.append(each_value)
                    else :
                        break

        previous_R_t = R_t
        P_t = [R_t[i] for i in new_solution]
        R_t = copy.deepcopy(P_t)

        ### Now, we develop the Q_t to get new R_t
        while len(R_t) != 2 * pop_size :
            parent1 = binary_tournament_selection(fronts,previous_R_t,P_t,crowding_distance_values,rank,True)
            parent2 = binary_tournament_selection(fronts,previous_R_t,P_t,crowding_distance_values,rank,True)

            (child1,child2) = crossover_and_mutation(P_t[parent1],P_t[parent2])
            R_t.append(child1)
            if len(R_t) == 2*pop_size :
                break
            else :
                R_t.append(child2)

        generation_no += 1

        # sum_in_objective_space = 0
        # for i in range(len(P_t)) :
        #     r = index_of(P_t[i],previous_R_t)
        #     sum_in_objective_space += crowding_distance_values[rank[r]][index_of(r,fronts[rank[r]])]

        # average_crowding_distance_objective_space.append(sum_in_objective_space/pop_size)

        # sum_in_data_space = 0
        # P_t_copy = copy.deepcopy(P_t)
        # P_t_copy.sort()
        # crowding_distance_data_space = [0 for i in range(len(P_t_copy))]
        # crowding_distance_data_space[0] = 4444444444444444
        # crowding_distance_data_space[len(P_t_copy)-1] = 4444444444444444
        # P_t_max = max(P_t_copy)
        # P_t_min = min(P_t_copy)

        # for i in range(1,len(P_t)-1) :
        #     crowding_distance_data_space[i] += (abs(P_t_copy[i+1] - P_t_copy[i]) + abs(P_t_copy[i] - P_t_copy[i-1]))/(P_t_max - P_t_min + 1)
        #     sum_in_data_space += crowding_distance_data_space[i]

        # average_crowding_distance_data_space.append(sum_in_data_space/pop_size-2)
        # no_of_generations.append(generation_no)

        ### WHILE LOOP ENDS HERE

    population = P_t
    # print("FINAL SOLUTION : \n",population,"\n\n",max(population),"  ",min(population))

    function1_values = [function1(population[i]) for i in range(0,pop_size)]
    function2_values = [function2(population[i]) for i in range(0,pop_size)]

    f1 = plt.figure(1)
    plt.xlabel('x^2', fontsize=15)
    plt.ylabel('(x-2)^2', fontsize=15)
    plt.scatter(function1_values, function2_values)
    plt.grid()
    plt.savefig("O.png")

    f2 = plt.figure(figsize=(6.4,2))
    plt.plot(population,[0 for i in range(pop_size)],'x')
    plt.grid()
    plt.savefig("D.png")

    # f3 = plt.figure(figsize=(12,7))
    # ax = f3.add_subplot(111)

    # plt.xlabel('No. of Generations (Computational Effort)', fontsize=15)
    # plt.ylabel('Average Crowding Distance', fontsize=15)
    # ax.plot(no_of_generations,average_crowding_distance_objective_space,color='blue',marker='o',markersize=4,linewidth=1)
    # ax.plot(no_of_generations,average_crowding_distance_data_space,color='orange',marker='o',markersize=4,linewidth=1)
    # plt.grid()
    # plt.show()
