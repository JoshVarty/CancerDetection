import os

def mergeResults(paths):

    averageRatio = 1/(len(paths))

    lookup = {}
    header = None

    for path in paths:
        with open(path, 'r') as file:
            #Read header
            header = file.readline()

            for line in file:
                splitLine = line.split(',')
                id = splitLine[0]
                score = splitLine[1]

                if id not in lookup:
                    lookup[id] = 0

                lookup[id] = lookup[id] + (averageRatio * score)

    # write our lookup to a file
    with open('mergedPredictions.csv', 'w') as file:
        #Write header
        file.write(header + os.linesep)

        for id, score in enumerate(lookup):
            file.write(id)
            file.write(',')
            file.write(score)
            file.write(os.linesep)


if __name__ == '__main__':
    fileNames = ['data/resnet50.csv', 'data/resnet101.csv', 'data/resnet152.csv', 'data/densenet121.csv']
    mergeResults(fileNames)