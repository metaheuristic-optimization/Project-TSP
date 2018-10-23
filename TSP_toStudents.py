"""
Author: David Ahern - R00002267
"""

import random
from Individual import *
import sys
import time
import math

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _selectedCandidateSelectionMethod, _selectedCrossoverMethod, _selectedMutationMethod):
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

        # User input variables
        self.candidateSelectionMethods = [self.randomSelection, self.rouletteWheel, self.bestAndSecondBest]
        self.selectedCandidateSelectionMethod = int(_selectedCandidateSelectionMethod)
        self.crossoverSelectionMethods = [self.uniformCrossover, self.cycleCrossover]
        self.selectedCrossoverMethod = int(_selectedCrossoverMethod)
        self.mutationMethods = [self.scrambleMutation, self.reciprocalExchangeMutation]
        self.selectedMutationMethod = int(_selectedMutationMethod)

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
        Roulette Wheel Selection Implementation
        """
        totalProbability = 0
        wheel = {}

        for i in range(0, len(self.matingPool)):
            probability = 1 / float(self.matingPool[i].fitness)
            self.matingPool[i].setSelectionProbability(probability)
            totalProbability += probability

        """
        Compute the wheel with both start and end points. 
        Points with the largest distance between their stand and end points have a better chance of being selected
        """
        total = 0
        for i in range(0, len(self.matingPool)):
            end = self.matingPool[i].selectionProbability + total
            wheel[i] = {"index": i, "start": total, "end": end}
            total = end

        """
        Spin the wheel to get the best candidate
        Generate 2 random points and compute the position they lay on the wheel
        """
        spinResultA = random.uniform(0, total)
        spinResultB = random.uniform(0, total)

        for i in wheel:
            if spinResultA >= wheel[i]["start"] and spinResultA < wheel[i]["end"]:
                indA = self.matingPool[wheel[i]["index"]]
            if spinResultB >= wheel[i]["start"] and spinResultB < wheel[i]["end"]:
                indB = self.matingPool[wheel[i]["index"]]

        return [indA, indB]

    def bestAndSecondBest(self):
        tmpPool = self.matingPool.copy()
        sortedPool = self.quickSort(tmpPool)

        return sortedPool[0], sortedPool[1]

    def quickSort(self, list):
        """
        Uses quick sort recursive algorithm to sort the list based on fitness
        """
        if len(list) == 0:
            return list

        left = []
        right = []
        equal = []
        pivot = list[0].fitness

        for ind in list:
            if ind.fitness > pivot:
                left.append(ind)
            elif ind.fitness == pivot:
                equal.append(ind)
            else:
                right.append(ind)

        return self.quickSort(left) + equal + self.quickSort(right)


    def uniformCrossover(self, indA, indB):
        """
        Uniform Crossover Implementation
        """
        child = []

        """
        Loop through the chromosome and populate a child. If the 50/50 random choice is True then we populate the current
        index of the child with the value from the first parent. If the random choice is False then we populate the current
        index of the child with a None value. Note as we are only generating one child we should not care about parent 2
        """
        for i in range(0, self.genSize):
            if random.choice([True, False]):
                child.append(indA.genes[i])
            else:
                child.append(None)

        """
        Loop through the 
        """
        for i in range(0, self.genSize):
            if not indB.genes[i] in child:
                nextFreeSlot = next(i for i,v in enumerate(child) if v == None)
                child[nextFreeSlot] = indB.genes[i]

        return child

    def cycleCrossover(self, indA, indB):
        """
        Cycle Crossover Implementation
        """
        flags = [False] * self.genSize
        child1 = [None] * self.genSize
        child2 = [None] * self.genSize
        cycles = []
        dictMap = {}

        """
        Build dictionary for fast lookup of indexes.
        """
        for i in range(0, self.genSize):
            dictMap[indA.genes[i]] = {'parent1': indA.genes[i], 'parent2': indB.genes[i], 'index': i}

        """
        Pre compute all the possible cycles. This makes generating the children much simpler
        """
        for i in range(0, self.genSize):
            tmpCycle = []

            # Make sure value is not already in another cycle to avoid duplication
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

        """
        Alternate cycles to generate the children
        """
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

        return child1

    def reciprocalExchangeMutation(self, ind):
        """
        Reciprocal Exchange Mutation implementation
        """

        """
        Check if mutation will take place based on the mutation rate
        """
        if not self.doMutation():
            return

        """
        Select 2 random indexes in the gene
        """
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        """
        Store the first gene in a temporary variable for later insertion
        """
        tmp = ind.genes[indexA]

        """
        Swap the two selected genes around
        """
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        self.updateBest(ind)

    def scrambleMutation(self, ind):
        """
        Scramble Mutation implementation
        """

        """
        Check if mutation will take place based on the mutation rate
        """
        if not self.doMutation():
            return

        """
        Select 2 random indexes. We need to make sure the first index is at least 1 less than the size of the array
        to allow the second index room. The second index needs to be after the first index so we can choose everything
        in between
        """
        indexStart = random.randint(0, self.genSize-2)
        indexEnd = random.randint(indexStart + 1, self.genSize-1)

        """
        Take a slice of the array between the two index and store it into a temporary variable
        """
        tmp = ind.genes[indexStart:indexEnd]

        """
        Shuffle the temporary array of values randomly
        """
        random.shuffle(tmp)

        """
        Re-insert the shuffled array back into the original genes array
        """
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
            [ind1, ind2] = self.candidateSelectionMethods[self.selectedCandidateSelectionMethod - 1]()
            child = self.crossoverSelectionMethods[self.selectedCrossoverMethod - 1](ind1, ind2)
            self.population[i].setGene(child)
            self.mutationMethods[self.selectedMutationMethod - 1](self.population[i])

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


"""
Presnt user with menu
"""
selectedCandidateSelectionMethod = input("Please select candidate selection method \n1) Random selection \n2) Roulette Wheel \n3) Best and second best\n")
selectedCrossoverMethod = input("Please select crossover method \n1) Uniform crossover\n2) Cycle crossover \n")
selectedMutationMethod = input("Please select mutation method \n1) Scramble mutation \n2) Reciprocal exchange mutation \n")

start_time = time.time()

ga = BasicTSP(sys.argv[1], 300, 0.1, 300, selectedCandidateSelectionMethod, selectedCrossoverMethod, selectedMutationMethod)

ga.search()

print("Process took %s seconds" % int(time.time() - start_time))
