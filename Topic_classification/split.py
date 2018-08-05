import os
import random
path="./TrainData/pre1.txt"

def SplitData(path):
    #dirs = os.listdir( path )
    #print(dirs)
    count=len(open(path,"rb").readlines())
    print(count)
    f10out = open("./DataTest/10-percent-output.txt", 'wb')
    f90out = open("./DataTrain/90-percent-output.txt", 'wb')
    fin =open(path,"rb")
    for line in fin:
        r = random.random()
        if r < 0.9:
            f90out.write(line)
        else:
            f10out.write(line)
    f10out.close()
    f90out.close()
    fin.close()
    cout1=len(open("./DataTest/10-percent-output.txt","rb").readlines())
    cout2=len(open("./DataTrain/90-percent-output.txt","rb").readlines())
    print(cout1)
    print(cout2)

SplitData(path)