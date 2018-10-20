

"""
Author:
file:
"""

import random
from Individual import *
import sys
import math
import itertools
import functools

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = _popSize
        self.genSize        = None
        self.mutationRate   = _mutationRate
        self.maxIterations  = _maxIterations
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}

        self.readInstance()
        self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())

    def updateBest(self, candidate):
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def rouletteWheel(self):
        """
        Your Roulette Wheel Selection Implementation
        """
        totalProbability = 0
        wheel = {}

        for i in range(0, len(self.matingPool)):
            probability = 1 / float(self.matingPool[i].fitness)
            self.matingPool[i].setSelectionProbability(probability)
            totalProbability += probability

        """
        Compute the wheel
        """
        total = 0
        for i in range(0, len(self.matingPool)):
            end = self.matingPool[i].selectionProbability + total
            wheel[i] = {"index": i, "start": total, "end": end}
            total = end

        """
        Spin the wheel to get best candidates
        """
        spinResultA = random.uniform(0, total)
        spinResultB = random.uniform(0, total)

        for i in wheel:
            if spinResultA >= wheel[i]["start"] and spinResultA < wheel[i]["end"]:
                indA = self.matingPool[wheel[i]["index"]]
            if spinResultB >= wheel[i]["start"] and spinResultB < wheel[i]["end"]:
                indB = self.matingPool[wheel[i]["index"]]

        return [indA, indB]

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation

        Uniform Order-based Crossover

        Week 5 17:41
        """
        child = []

        for i in range(0, self.genSize):
            if random.choice([True, False]):
                child.append(indA.genes[i])
            else:
                child.append(None)

        for i in range(0, self.genSize):
            if not indB.genes[i] in child:
                nextFreeSlot = next(i for i,v in enumerate(child) if v == None)
                child[nextFreeSlot] = indB.genes[i]

        return child

    def cycleCrossover(self, indA, indB):
        """
        Your Cycle Crossover Implementation
        """
        flags = [False] * self.genSize
        child1 = [None] * self.genSize
        child2 = [None] * self.genSize
        cycles = []
        dictMap = {}

        print(indA.genes)
        print(indB.genes)

        # Build dictionary for fast lookup of indexes
        for i in range(0, self.genSize):
            dictMap[indA.genes[i]] = {'parent1': indA.genes[i], 'parent2': indB.genes[i], 'index': i}

        # Compute all the cycles
        for i in range(0, self.genSize):
            tmpCycle = []

            # Make sure value is not already in another cycle
            if not flags[i]:
                cycleStart = indA.genes[i]
                tempPair = dictMap[indA.genes[i]]
                tmpCycle.append(tempPair)
                flags[tempPair['index']] = True

                while not tempPair['parent2'] == cycleStart:
                    tempPair = dictMap[tempPair['parent2']]
                    flags[tempPair['index']] = True
                    tmpCycle.append(tempPair)

                cycles.append(tmpCycle)


        print(cycles)

        # Alternate cycles to generate the children
        counter = 0
        for cycle in cycles:
            for pair in cycle:
                if counter % 2 == 0:
                    child1[pair['index']] = pair['parent1']
                    child2[pair['index']] = pair['parent2']
                else:
                    child1[pair['index']] = pair['parent2']
                    child2[pair['index']] = pair['parent1']
            counter += 1

        print(child1)
        print(child2)

        pass

    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation

        Week 5 - 11:20
        """
        if not self.doMutation():
            return

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def scrambleMutation(self, ind):
        """
        Your Scramble Mutation implementation

        Week 5 - 11:30
        """
        if not self.doMutation():
            return

        indexStart = random.randint(0, self.genSize-2)
        indexEnd = random.randint(indexStart + 1, self.genSize-1)

        tmp = ind.genes[indexStart:indexEnd]

        random.shuffle(tmp)
        ind.genes[indexStart:indexEnd] = tmp

        ind.computeFitness()
        self.updateBest(ind)

        pass

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if not self.doMutation():
            return

        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def doMutation(self):
        """
        Helper function to decide if a mutation will take place based on the mutation rate

        :return:  Boolean
        """
        return random.random() > self.mutationRate

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )

    def newGeneration(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            [ind1, ind2] = self.rouletteWheel()
            child = self.cycleCrossover(ind1, ind2)
            self.population[i].setGene(child)
            self.reciprocalExchangeMutation(self.population[i])

    def GAStep(self):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        self.newGeneration()

    def search(self):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        while self.iteration < self.maxIterations:
            self.GAStep()
            self.iteration += 1

        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness())

if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)


problem_file = sys.argv[1]

ga = BasicTSP(sys.argv[1], 300, 0.1, 300)
ga.search()