import csv
import ID3


### CAR EXAMPLE ###
featureNames = ['buying','maint','doors','persons','lug_boot','safety']
uniqueLabels = ['unacc', 'acc', 'good', 'vgood']

data = []
labels = []

with open('DecisionTrees\HW1Data\car\\train.csv', mode='r') as file:
    csvReader = csv.reader(file)
    count = 0
    for row in csvReader:
        count += 1
        #separate targets from predictors
        data.append(row[:6])
        labels.append(row[6])
    #print(f'Processed {count} lines.\n')

print("Using data from car.zip\n")

# instantiate DecisionTreeClassifier
ID3ContextStandard = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels)
ID3ContextME = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.MAJORITY_ERROR)
ID3ContextGI = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.GINI_INDEX)

# run algorithm to build my tree
for i in range(6):
    print("maxDepth=", i+1)
    ID3ContextStandard.ID3(i + 1)
    ID3ContextStandard.testData("DecisionTrees\HW1Data\car\\train.csv", 6, "IG\t")
    ID3ContextME.ID3(i + 1)
    ID3ContextME.testData("DecisionTrees\HW1Data\car\\train.csv", 6, "ME\t")
    ID3ContextGI.ID3(i + 1)
    ID3ContextGI.testData("DecisionTrees\HW1Data\car\\train.csv", 6, "GI\t")
    print("\n")

print("maxDepth=no_limit")
ID3ContextStandard.ID3()
ID3ContextStandard.testData("DecisionTrees\HW1Data\car\\train.csv", 6, "IG\t")
ID3ContextME.ID3()
ID3ContextME.testData("DecisionTrees\HW1Data\car\\train.csv", 6, "ME\t")
ID3ContextGI.ID3()
ID3ContextGI.testData("DecisionTrees\HW1Data\car\\train.csv", 6, "GI\t")
print("\n")

### BANK EXAMPLE ###

featureNames = ['age','job','marital','education','default','balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
uniqueLabels = ['yes', 'no']

data = []
labels = []

with open('DecisionTrees\HW1Data\\bank\\train.csv', mode='r') as file:
    csvReader = csv.reader(file)
    count = 0
    for row in csvReader:
        count += 1
        #separate targets from predictors
        data.append(row[:16])
        labels.append(row[16])
    #print(f'\nProcessed {count} lines.\n')

print("Using data from bank.zip\n")

# instantiate DecisionTreeClassifier
ID3ContextStandard = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels)
ID3ContextME = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.MAJORITY_ERROR)
ID3ContextGI = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.GINI_INDEX)

# run algorithm to build my tree
ID3ContextStandard.ID3()
# ID3ContextStandard.printTree()

ID3ContextME.ID3()
# ID3ContextME.printTree()

ID3ContextGI.ID3()
# ID3ContextGI.printTree()
