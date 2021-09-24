import math
from collections import deque

ENTROPY = 0
MAJORITY_ERROR = 1
GINI_INDEX = 2

class Node:
    def __init__(self):
        # feature being examined
        self.value = None
        self.depth = 0
        self.next = None
        # children (branches that represent attribute values)
        self.children = None

# sets up necessary information to run the algorithm
class Classifier:
    def __init__(self, S, featureNames, labels, uniqueLabels, method=ENTROPY, maxDepth=-1):
        self.maxDepth = maxDepth
        self.root = None

        self.method = method
        self.S = S
        self.featureNames = featureNames
        # outcomes
        self.labels = labels
        self.uniqueLabels = uniqueLabels

        # array that counts how many labels are in each category
        self.labelCategoriesCount = [list(labels).count(i) for i in self.uniqueLabels]
        # calculates initial entropy
        self.entropy = self.getEntropy([i for i in range(len(self.labels))])

    def getEntropy(self, sIndexes):
        if self.method == ENTROPY :
          return self.getEntropyStandard(sIndexes)
        elif self.method == MAJORITY_ERROR:
          return self.getMajorityError(sIndexes)
        elif self.method == GINI_INDEX:
          return self.getGiniIndex(sIndexes)
        else:
          print("yoU bROkE iT.") # should be unreachable :)
          return

# METHODS FOR CALCULATING ENTROPY/PURITY #
    def getEntropyStandard(self, sIndexes):
        # sorted labels by index
        labels = [self.labels[i] for i in sIndexes]
        # count number of instances in each unique label
        lblCount = [labels.count(i) for i in self.uniqueLabels]
        return sum([-count / len(sIndexes) * math.log(count / len(sIndexes), 2) if count else 0 for count in lblCount])
  
    def getMajorityError(self, sIndexes):
        # sorted labels by index
        labels = [self.labels[i] for i in sIndexes]
        # count number of instances in each unique label
        lblCount = [labels.count(i) for i in self.uniqueLabels]
        majorityLabel = lblCount.index(max(lblCount))
        return (len(sIndexes)-lblCount[majorityLabel])/len(sIndexes)
  
    def getGiniIndex(self, sIndexes):
        # sorted labels by index
        labels = [self.labels[i] for i in sIndexes]
        # count number of instances in each unique label
        lblCount = [labels.count(i) for i in self.uniqueLabels]
        return 1 - sum([(count/len(sIndexes))**2 if count else 0 for count in lblCount])
# END METHODS #############################

    # calculate information gain using the original method discussed in lecture (expected reduction in entropy)
    def calcGain(self, sIndexes, fid):
        initEntropy = self.getEntropy(sIndexes)

        sFeatures = [self.S[i][fid] for i in sIndexes]
        uniqueVals = list(set(sFeatures))
        valCounts = [sFeatures.count(i) for i in uniqueVals]

        uniqueIndexes = [
            [sIndexes[i]
            for i, x in enumerate(sFeatures)
            if x == y]
            for y in uniqueVals
        ]
        
        newEntropy = sum([valCounts / len(sIndexes) * self.getEntropy(uniqueIndexes) for valCounts, uniqueIndexes in zip(valCounts, uniqueIndexes)])
        return initEntropy - newEntropy

    # find the attribute that maximizes information gain
    def calcMaxGain(self, sIndexes, featureIndexes):
        # get the entropy for each feature
        featureEntropies = [self.calcGain(sIndexes, fid) for fid in featureIndexes]
        maxGainIndex = featureIndexes[featureEntropies.index(max(featureEntropies))]

        return self.featureNames[maxGainIndex], maxGainIndex

    # overhead for the recursive call to ID3 algorithm
    def ID3(self):
        sIndexes = [i for i in range(len(self.S))]
        featureIndexes = [i for i in range(len(self.featureNames))]
        self.root = self.ID3Recurse(sIndexes, featureIndexes, self.root)

    # recursive id3 function
    def ID3Recurse(self, sIndexes, featureIndexes, node):
        if not node:
            node = Node() 
            node.depth = 0

        nodeLabels = [self.labels[i] for i in sIndexes]
        if len(set(nodeLabels)) == 1: # all labels the same (pure)
             node.value = self.labels[sIndexes[0]]
             return node

        if len(featureIndexes) == 0:  # no remaining features
            node.value = max(set(nodeLabels), key=nodeLabels.count)  # return 'most probable' leaf
            return node

        # otherwise...

        # choose the feature that maximizes the information gain
        maxName, maxIndex = self.calcMaxGain(sIndexes, featureIndexes)
        node.value = maxName
        node.children = []
        # values of the "best" feature
        maxFeatureVals = list(set([self.S[i][maxIndex] for i in sIndexes]))

        for val in maxFeatureVals:
            child = Node()

            # add branch (feature)
            child.value = val  
            child.depth = node.depth + 1
            node.children.append(child) 

            childIndexes = [i for i in sIndexes if self.S[i][maxIndex] == val]
            
            if not childIndexes:  
                child.next = max(set(nodeLabels), key=nodeLabels.count) # the next child is set to the node with the most labels
            else:
                if featureIndexes and maxIndex in featureIndexes:
                    featureIndexes.pop(featureIndexes.index(maxIndex)) 
                
                if self.maxDepth == -1 or child.depth < self.maxDepth :
                    # recursively call the algorithm
                    child.next = self.ID3Recurse(childIndexes, featureIndexes, child.next)

        return node

    def printTree(self):
        print("Tree method =", self.method)
        if not self.root:
            print("ERROR: No tree")
            return
        nodes = deque()
        nodes.append(self.root)
        while len(nodes) > 0:
            node = nodes.popleft()
            print(node.value)
            if node.children:
                for child in node.children:
                    print('(', child.value, ')')
                    nodes.append(child.next)
            elif node.next:
                print("Next node: ", node.next)
        print("\n\n")