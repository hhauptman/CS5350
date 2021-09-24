import csv
import ID3

### CAR EXAMPLE ###
featureNames = ['buying','maint','doors','persons','lug_boot','safety']
uniqueLabels = ['unacc', 'acc', 'good', 'vgood']

data = []
labels = []

with open('HW1Data/car/train.csv', mode='r') as file:
    csvReader = csv.reader(file)
    count = 0
    for row in csvReader:
        count += 1
        #separate targets from predictors
        data.append(row[:6])
        labels.append(row[6])
    print(f'Processed {count} lines.')

# instantiate DecisionTreeClassifier
ID3ContextStandard = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels)
ID3ContextME = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.MAJORITY_ERROR)
ID3ContextGI = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.GINI_INDEX)

print("System entropy {:.4f}".format(ID3ContextStandard.entropy))
print("System entropy {:.4f}".format(ID3ContextME.entropy))
print("System entropy {:.4f}".format(ID3ContextGI.entropy))
print("\n")

# run algorithm to build my tree
ID3ContextStandard.ID3()
ID3ContextStandard.printTree()

ID3ContextME.ID3()
ID3ContextME.printTree()

ID3ContextGI.ID3()
ID3ContextGI.printTree()


### BANK EXAMPLE ###

featureNames = ['age','job','marital','education','default','balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
uniqueLabels = ['yes', 'no']

data = []
labels = []

with open('HW1Data/bank/train.csv', mode='r') as file:
    csvReader = csv.reader(file)
    count = 0
    for row in csvReader:
        count += 1
        #separate targets from predictors
        data.append(row[:16])
        labels.append(row[16])
    print(f'Processed {count} lines.')

# instantiate DecisionTreeClassifier
ID3ContextStandard = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels)
ID3ContextME = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.MAJORITY_ERROR)
ID3ContextGI = ID3.Classifier(S=data, featureNames=featureNames, labels=labels, uniqueLabels=uniqueLabels, method=ID3.GINI_INDEX)

print("System entropy {:.4f}".format(ID3ContextStandard.entropy))
print("System entropy {:.4f}".format(ID3ContextME.entropy))
print("System entropy {:.4f}".format(ID3ContextGI.entropy))
print("\n")

# run algorithm to build my tree
ID3ContextStandard.ID3()
# ID3ContextStandard.printTree()

ID3ContextME.ID3()
# ID3ContextME.printTree()

ID3ContextGI.ID3()
# ID3ContextGI.printTree()