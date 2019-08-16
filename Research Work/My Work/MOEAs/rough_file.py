import math
import random
import copy
import matplotlib.pyplot as plt

def index_of(a,list): #returns index of "a" in "list" , else "-1"
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(front, values):
    sorted_list = []
    while(len(sorted_list) != len(front)):
        sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

def avg_crowding_distance(values1,values2,f) :
    distance = [0 for i in range(4)]

    distance[0] = 444 ### BIG NUMBER
    distance[len(f) - 1] = 444  ### BIG NUMBER

    for k in range(1,len(f)-1):
        distance[k] = distance[k] + (values1[k+1] - values1[k-1])/(max(values1)-min(values1) + 1)
    for k in range(1,len(f)-1):
        distance[len(f)-k-1] = distance[len(f)-k-1] + (values2[k+1] - values2[k-1])/(max(values2)-min(values2) + 1)

    sum_in_objective_space = 0
    for i in range(1,len(f)-1) :
        sum_in_objective_space += distance[i]

    return sum_in_objective_space/4

l1 = [[0.1,0.2,3.8,3.9],[1,2,3,4],[1.8,2.0,2.2,2.4]]
l2 = [[3.9,3.8,0.2,0.1],[4,3,2,1],[2.2,2.0,1.8,1.6]]

for i in range(3):
    print(avg_crowding_distance(l1[i],l2[i],[1,2,3,4]))

# f1 = plt.figure(1)
# plt.xlim(-1,5)
# plt.ylim(-1,5)
# plt.grid()
# plt.scatter(l1[0],l2[0])
#
# f2 = plt.figure(2)
# plt.xlim(-1,5)
# plt.ylim(-1,5)
# plt.grid()
# plt.scatter(l1[1],l2[1])
#
# f3 = plt.figure(3)
# plt.xlim(-1,5)
# plt.ylim(-1,5)
# plt.grid()
# plt.scatter(l1[2],l2[2])
#
#
# plt.show()
