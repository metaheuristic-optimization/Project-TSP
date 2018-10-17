

"""
Author:
file:
"""

import random
from Individual import *
import sys
import math

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
        pass

    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation

        Uniform Order-based Crossover

        Week 5 17:41
        """
        childA = []
        childB = []

        for i in range(0, self.genSize-1):
            choice = random.choice([True, False])
            childA += [indA.genes[i] if choice else None]
            childB += [indB.genes[i] if choice else None]


        print(childA)
        print(childB)


        """
        totalIndexes = random.randint(0, self.genSize-1)
        selectedIndexes = random.sample(list(enumerate(indA.genes)), 1)

        tmpChildA = {}
        tmpChildB = {}

        for i, _ in selectedIndexes:
            tmpChildA[indA.genes[i]] = i
            tmpChildB[indA.genes[i]] = i

        print(tmpChildA)

        for i in range(0, self.genSize):
            if not indB.genes[i] in tmpChildA:
                tmpChildA[indB.genes[i]] = i
            if not indA.genes[i] in tmpChildB:
                tmpChildA[indA.genes[i]] = i

        print('==========================================')
        print(tmpChildA)

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print(indA.genes)
        print('++++++++++++++++++++++++++++++++++++++++++')
        print([tmpChildA.get(i) for i in range(1, max(tmpChildA) + 1)])
        """
        pass

    def cycleCrossover(self, indA, indB):
        """
        Your Cycle Crossover Implementation
        """
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
            [ind1, ind2] = self.randomSelection()
            child = self.uniformCrossover(ind1, ind2)
            self.population[i].setGene(child)
            self.scrambleMutation(self.population[i])

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

ga = BasicTSP(sys.argv[1], 300, 0.1, 500)
ga.search()