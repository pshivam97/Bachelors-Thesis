# TAKEN FROM  :  https://gist.github.com/turbofart/3428880

import math
import random
import matplotlib.pyplot as plt
import copy

class City:
   def __init__(self, x=None, y=None):
      self.x = None
      self.y = None
      if x is not None:
         self.x = x
      else:
         self.x = int(random.random() * 200)
      if y is not None:
         self.y = y
      else:
         self.y = int(random.random() * 200)

   def getX(self):
      return self.x

   def getY(self):
      return self.y

   def distanceTo(self, city):
      xDistance = abs(self.getX() - city.getX())
      yDistance = abs(self.getY() - city.getY())
      distance = math.sqrt( (xDistance*xDistance) + (yDistance*yDistance) )
      return distance

   def __repr__(self):
      return str(self.getX()) + ", " + str(self.getY())


class TourManager:
   destinationCities = []

   def addCity(self, city):
      self.destinationCities.append(city)

   def getCity(self, index):
      return self.destinationCities[index]

   def numberOfCities(self):
      return len(self.destinationCities)


class Tour:

   def __init__(self, tourmanager, tour=None):
      self.tourmanager = tourmanager
      self.tour = []
      self.fitness = 0.0
      self.distance = 0

      if tour is not None:
         self.tour = tour
      else:
         for i in range(0, self.tourmanager.numberOfCities()):
            self.tour.append(None)

   def __len__(self):
      return len(self.tour)

   def __getitem__(self, index):
      return self.tour[index]

   def __setitem__(self, key, value):
      self.tour[key] = value

   def __repr__(self):
      geneString = "|"
      for i in range(0, self.tourSize()):
         geneString += str(self.getCity(i)) + "|"
      return geneString

   def generateIndividual(self):
      global total_fitness_count
      global single_gen_fitness_count
      global single_gen_fitness
      global single_gen_fitness_sum

      for cityIndex in range(0, self.tourmanager.numberOfCities()):
         self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
      random.shuffle(self.tour)
      self.fitness = 1/float(self.getDistance())

   def getCity(self, tourPosition):
      return self.tour[tourPosition]

   def setCity(self, tourPosition, city):
      self.tour[tourPosition] = city
      self.fitness = 0.0
      self.distance = 0

   def getFitness(self,FitnessProportionate):

      global single_gen_fitness_countF
      global total_fitness_countF
      global single_gen_fitness_sumF
      global single_gen_fitnessF
      global single_gen_fitness_countT
      global total_fitness_countT
      global single_gen_fitness_sumT
      global single_gen_fitnessT

      if self.fitness == 0:

         if FitnessProportionate :
             self.fitness = 1/float(self.getDistance())
             single_gen_fitnessF.append(self.fitness)

             single_gen_fitness_countF += 1
             total_fitness_countF += 1
             single_gen_fitness_sumF += self.fitness

             if single_gen_fitness_countF == population_size :
                 no_of_fitness_evaluationsF.append(total_fitness_countF)
                 single_gen_fitness_countF = 0
                 average_fitnessF.append(single_gen_fitness_sumF/population_size)
                 single_gen_fitness_sumF = 0
                 fittest_individualsF.append(max(single_gen_fitnessF))
                 del single_gen_fitnessF[:]
         else :
             self.fitness = 1/float(self.getDistance())
             single_gen_fitnessT.append(self.fitness)

             single_gen_fitness_countT += 1
             total_fitness_countT += 1
             single_gen_fitness_sumT += self.fitness

             if single_gen_fitness_countT == population_size :
                 no_of_fitness_evaluationsT.append(total_fitness_countT)
                 single_gen_fitness_countT = 0
                 average_fitnessT.append(single_gen_fitness_sumT/population_size)
                 single_gen_fitness_sumT = 0
                 fittest_individualsT.append(max(single_gen_fitnessT))
                 del single_gen_fitnessT[:]

      return self.fitness

   def getDistance(self):
      if self.distance == 0:
         tourDistance = 0
         for cityIndex in range(0, self.tourSize()):
            fromCity = self.getCity(cityIndex)
            destinationCity = None
            if cityIndex+1 < self.tourSize():
               destinationCity = self.getCity(cityIndex+1)
            else:
               destinationCity = self.getCity(0)
            tourDistance += fromCity.distanceTo(destinationCity)
         self.distance = tourDistance
      return self.distance

   def tourSize(self):
      return len(self.tour)

   def containsCity(self, city):
      return city in self.tour


class Population:
   def __init__(self, tourmanager, populationSize, initialise):
      self.tours = []
      self.fitness_values = []
      for i in range(0, populationSize):
         self.tours.append(None)

      if initialise:
         for i in range(0, populationSize):
            newTour = Tour(tourmanager)
            newTour.generateIndividual()
            self.saveTour(i, newTour)
            self.fitness_values.append(newTour.fitness)

   def __setitem__(self, key, value):
      self.tours[key] = value

   def __getitem__(self, index):
      return self.tours[index]

   def saveTour(self, index, tour):
      self.tours[index] = tour

   def getTour(self, index):
      return self.tours[index]

   def getFittest(self,FitnessProportionate):
      fittest = self.tours[0]
      for i in range(1, self.populationSize()):
         if self.getTour(i).getFitness(FitnessProportionate) > fittest.getFitness(FitnessProportionate):
            fittest = self.getTour(i)
      return fittest

   def populationSize(self):
      return len(self.tours)

class GA:
   def __init__(self, tourmanager):
      self.tourmanager = tourmanager
      self.mutationRate = 0.015
      self.tournamentSize = 5
      self.elitism = True

   def evolvePopulation(self, pop, FitnessProportionate):

      newPopulation = Population(self.tourmanager, pop.populationSize(), False)
      elitismOffset = 0
      if self.elitism:
         fittest = pop.getFittest(FitnessProportionate)
         fittest.fitness = 0.0

         newPopulation.saveTour(0, fittest)
         newPopulation.fitness_values.append(1/float(fittest.getDistance()))
         elitismOffset = 1

      for i in range(elitismOffset, newPopulation.populationSize()):
         if FitnessProportionate :
            parent1 = self.fitnessProportionateSelection(pop)
            parent2 = self.fitnessProportionateSelection(pop)

         else :
            parent1 = self.tournamentSelection(pop,False)
            parent2 = self.tournamentSelection(pop,False)
         child = self.crossover(parent1, parent2)
         # childt = self.crossover(parent1t, parent2t)
         newPopulation.saveTour(i, child)
         # newPopulation.fitness_values.append(1/float(newPopulation.getTour(i).getDistance()))

      for i in range(elitismOffset, newPopulation.populationSize()):
         self.mutate(newPopulation.getTour(i))
         newPopulation.fitness_values.append(1/float(newPopulation.getTour(i).getDistance()))

      # print(newPopulation.getTour(0))
      return newPopulation



   def crossover(self, parent1, parent2):
      child = Tour(self.tourmanager)

      startPos = int(random.random() * parent1.tourSize())
      endPos = int(random.random() * parent1.tourSize())

      for i in range(0, child.tourSize()):
         if startPos < endPos and i > startPos and i < endPos:
            child.setCity(i, parent1.getCity(i))
         elif startPos > endPos:
            if not (i < startPos and i > endPos):
               child.setCity(i, parent1.getCity(i))

      for i in range(0, parent2.tourSize()):
         if not child.containsCity(parent2.getCity(i)):
            for ii in range(0, child.tourSize()):
               if child.getCity(ii) == None:
                  child.setCity(ii, parent2.getCity(i))
                  break

      return child

   def mutate(self, tour):
      for tourPos1 in range(0, tour.tourSize()):
         if random.random() < self.mutationRate:
            tourPos2 = int(tour.tourSize() * random.random())

            city1 = tour.getCity(tourPos1)
            city2 = tour.getCity(tourPos2)

            tour.setCity(tourPos2, city1)
            tour.setCity(tourPos1, city2)

   def tournamentSelection(self, pop, flag):
      tournament = Population(self.tourmanager, self.tournamentSize, False)
      for i in range(0, self.tournamentSize):
         randomId = int(random.random() * pop.populationSize())
         tournament.saveTour(i, pop.getTour(randomId))
      fittest = tournament.getFittest(False)
      return fittest

   def fitnessProportionateSelection(self, pop) :

       random_number = random.random()
       probabilities_of_population = [pop.fitness_values[i]/sum(pop.fitness_values) for i in range(len(pop.fitness_values))]
       cumulative_probabilities = []
       cum_sum = 0

       for i in probabilities_of_population :
           cum_sum += i
           cumulative_probabilities.append(cum_sum)

       for i in range(len(cumulative_probabilities)) :
           if random_number <= cumulative_probabilities[i] :
               return pop.getTour(i)


if __name__ == '__main__':

   tourmanager = TourManager()

   # Create and add our cities
   city = City(60, 200)
   tourmanager.addCity(city)
   city2 = City(180, 200)
   tourmanager.addCity(city2)
   city3 = City(80, 180)
   tourmanager.addCity(city3)
   city4 = City(140, 180)
   tourmanager.addCity(city4)
   city5 = City(20, 160)
   tourmanager.addCity(city5)
   city6 = City(100, 160)
   tourmanager.addCity(city6)
   city7 = City(200, 160)
   tourmanager.addCity(city7)
   city8 = City(140, 140)
   tourmanager.addCity(city8)
   city9 = City(40, 120)
   tourmanager.addCity(city9)
   city10 = City(100, 120)
   tourmanager.addCity(city10)
   city11 = City(180, 100)
   tourmanager.addCity(city11)
   city12 = City(60, 80)
   tourmanager.addCity(city12)
   city13 = City(120, 80)
   tourmanager.addCity(city13)
   city14 = City(180, 60)
   tourmanager.addCity(city14)
   city15 = City(20, 40)
   tourmanager.addCity(city15)
   city16 = City(100, 40)
   tourmanager.addCity(city16)
   city17 = City(200, 40)
   tourmanager.addCity(city17)
   city18 = City(20, 20)
   tourmanager.addCity(city18)
   city19 = City(60, 20)
   tourmanager.addCity(city19)
   city20 = City(160, 20)
   tourmanager.addCity(city20)

   runs = 5
   total_run_of_algorithm = runs
   population_size = 50

   avg_fitness_zipped_listF = list()
   best_fitness_zipped_listF = list()
   avg_fitness_zipped_listT = list()
   best_fitness_zipped_listT = list()

   while runs != 0 :
   # THIS LOOP RUNS FOR THE USER-SPECIFIED AMOUNT OF TIMES (eg:- If runs = 5, then it runs 5 times)
       average_fitnessF = list()
       no_of_fitness_evaluationsF = list()
       fittest_individualsF = list()

       single_gen_fitness_countF = 0
       total_fitness_countF = 0
       single_gen_fitness_sumF = 0
       single_gen_fitnessF = list()

   # Initialize population
       initial_popF = Population(tourmanager, population_size, True);
       initial_popT = initial_popF

       # print("Initial distance: \n",f," ",str(1/float(pop.getFittest().getDistance()))," | ",f," ",fittest_individuals[f])

   # Evolve population for 50 generations
       ga = GA(tourmanager)

       for i in range(50):
           initial_popF = ga.evolvePopulation(initial_popF,True)
       #     # print(i+1," ",1/float(pop.getFittest().getDistance())," | ",i+1," ",fittest_individuals[i+1])
       #     # f += 1
       #
       avg_fitness_zipped_listF.append(average_fitnessF)
       best_fitness_zipped_listF.append(fittest_individualsF)

       # del average_fitness[:]
       # del fittest_individuals[:]
       # del no_of_fitness_evaluations[:]
       # single_gen_fitness_count = 0
       # total_fitness_count = 0
       # single_gen_fitness_sum = 0
       # del single_gen_fitness[:]

       average_fitnessT = list()
       no_of_fitness_evaluationsT = list()
       fittest_individualsT = list()

       single_gen_fitness_countT = 0
       total_fitness_countT = 0
       single_gen_fitness_sumT = 0
       single_gen_fitnessT = list()

       for i in range(50):
           initial_popT = ga.evolvePopulation(initial_popT,False)

       avg_fitness_zipped_listT.append(average_fitnessT)
       best_fitness_zipped_listT.append(fittest_individualsT)

   # Print final results
       # print("Finished")
       # print("Final distance: \n",str(1/float(pop.getFittest().getDistance()))," | ",f," ",fittest_individuals[50])
       # print("Solution:",pop.getFittest())
       runs -= 1

   fig = plt.figure(figsize=(12,7))
   ax = fig.add_subplot(111)

   all_runs_avg_fitnessF = [(1/total_run_of_algorithm)*sum(i) for i in zip(*avg_fitness_zipped_listF)]
   all_runs_best_fitnessF = [(1/total_run_of_algorithm)*sum(i) for i in zip(*best_fitness_zipped_listF)]
   all_runs_avg_fitnessT = [(1/total_run_of_algorithm)*sum(i) for i in zip(*avg_fitness_zipped_listT)]
   all_runs_best_fitnessT = [(1/total_run_of_algorithm)*sum(i) for i in zip(*best_fitness_zipped_listT)]

   # Curves for FitnessProportionate method
   ax.plot(no_of_fitness_evaluationsF, all_runs_avg_fitnessF,color='blue', marker='o', linewidth=1, markersize=6,label="Average Fitness (Fitness Proportionate)")
   ax.plot(no_of_fitness_evaluationsF, all_runs_best_fitnessF,color='orange', marker='o', linewidth=1, markersize=6,label="Best Fitness (Fitness Proportionate)")

   # Curves for Tournament Selection method
   ax.plot(no_of_fitness_evaluationsT, all_runs_avg_fitnessT,color='red', marker='x', linewidth=1, markersize=6,label="Average Fitness (Tournament)")
   ax.plot(no_of_fitness_evaluationsT, all_runs_best_fitnessT,color='green', marker='x', linewidth=1, markersize=6,label="Best Fitness (Tournament)")

   plt.xlabel('No. of Fitness Evaluations (Computational Effort)', fontsize=15)
   plt.ylabel('Fitness Values', fontsize=15)
   plt.legend()
   plt.ylim(0.00035, 0.001)
   plt.grid()
   plt.show()
