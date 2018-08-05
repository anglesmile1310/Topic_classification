import json
from bs4 import BeautifulSoup
import re
import data_helpers
import time
def normalize(text):
    text=BeautifulSoup(text).get_text()#loai tag html
    text=re.sub(r'[\n+]', ' ',text)#loai dau xuong dong
    text=re.sub(r'[^\w]', ' ',text)
    text = re.sub('\s+', ' ', text)  # loai khoang trong
    return text

def Predata(n):
    for i in range(1,n+1):
        start = time.time()
        with open("./rawdata/"+str(i)+".json","r") as f:
            datas=json.load(f)
        with open("./TrainData/pre"+str(i)+".txt",'wb') as g:
            for data in datas:
                g.write((BeautifulSoup(data["content"]).get_text())+"\n")
        
            #with open("./predata/pre"+str(i)+".txt",'wb') as f:
        #f=open("rawdata/"+str(i)+".json","rb")
        #datas=json.load(f)
        train= open("./DataTrain/"+str(i)+".txt","wb")
        test=  open("./DataTest/"+str(i)+".txt","wb")
        for data in datas:
            if BeautifulSoup(data["content"]).get_text().strip()!="":
                r=random.random()
                if r<0.9:    
                    train.write(data_helpers.remove_Stopword(data_helpers.tokenize(data_helpers.normalize_Text( normalize(data["content"]))))+"\n")
                else:
                    test.write(data_helpers.remove_Stopword(data_helpers.tokenize(data_helpers.normalize_Text( normalize(data["content"]))))+"\n")
        end=time.time()
        print(end-start)
        #f.close()
    
Predata(int(6))
#chu y sau khi tien xu ly xong thi phai chia lai du lieu cho Test va Train
