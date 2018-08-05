import numpy as np
import re
from pyvi import ViTokenizer
import pandas as pd
import string
import time
from bs4 import BeautifulSoup
import itertools
from collections import Counter

# data dict_abbreviation
filename = './data/dict/dict_abbreviation.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_abbreviation = data['abbreviation']
list_converts = data['convert']
#data stopword
FileStopword='./data/dict/stopwords.csv'
DataSW=pd.read_csv(FileStopword,sep="\t",encoding='utf-8')
list_stopwords=DataSW['stopwords']

#hàm này đọc dataset raw và tách chuỗi câu
def readdata(path):
    with open(path, 'r',encoding="UTF-8") as f:
        rawdata = f.read().split("\n") #Hàm này sẽ tách chuỗi bởi các ký tự \n.(tách chuỗi theo dòng)
    return [rawdata[i] for i in range(len(rawdata))]#trả về list các đoạn văn bản trong một chủ đề


def clean_data(comment):
    # loai link lien ket
    comment = re.sub(r'\shttps?:\/\/[^\s]*\s+|^https?:\/\/[^\s]*\s+|https?:\/\/[^\s]*$', ' link_spam ', comment)
    #chuyển hết link trong comment thành "link_spam"
    return comment

def convert_Abbreviation(comment):
    comment = re.sub('\s+', " ", comment)
    for i in range(len(list_converts)):
        abbreviation = '(\s' + list_abbreviation[i] + '\s)|(^' + list_abbreviation[i] + '\s)|(\s' \
                       + list_abbreviation[i] + '$)'
        convert = ' ' + str(list_converts[i]) + ' '
        comment = re.sub(abbreviation, convert, comment)

    return comment

def remove_Stopword(comment):
    re_comment = []
    words = comment.split()
    for word in words:
        if (not word.isnumeric()) and len(word) > 1 and word not in list_stopwords:
            re_comment.append(word)
    comment = ' '.join(re_comment)
    return comment


def tokenize(comment):
    text_token = ViTokenizer.tokenize(comment)
    return text_token

def normalize_Text(comment):
    comment = comment.encode().decode()
    comment = comment.lower()


    #thay gia tien bang text
    moneytag = [u'k', u'đ', u'ngàn', u'nghìn', u'usd', u'tr', u'củ', u'triệu', u'yên']
    for money in moneytag:
        comment = re.sub('(^\d*([,.]?\d+)+\s*' + money + ')|(' + '\s\d*([,.]?\d+)+\s*' + money + ')', ' monney ',
                         comment)
    comment = re.sub('(^\d+\s*\$)|(\s\d+\s*\$)', ' monney ', comment)
    comment = re.sub('(^\$\d+\s*)|(\s\$\d+\s*\$)', ' monney ', comment)
    # loai dau cau: nhuoc diem bi vo cau truc: vd; km/h. V-NAND
    listpunctuation = string.punctuation
    for i in listpunctuation:
        comment = comment.replace(i, ' ')
    # thay thong so bang specifications
    comment = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', comment)
    comment = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', comment)
    # thay thong so bang text lan 2
    comment = re.sub('^(\d+[a-z]+)([a-z]*\d*)*\s|\s\d+[a-z]+([a-z]*\d*)*\s|\s(\d+[a-z]+)([a-z]*\d*)*$', ' ', comment)
    comment = re.sub('^([a-z]+\d+)([a-z]*\d*)*\s|\s[a-z]+\d+([a-z]*\d*)*\s|\s([a-z]+\d+)([a-z]*\d*)*$', ' ', comment)
    # xu ly lay am tiet
    comment = re.sub(r'(\D)\1+', r'\1', comment)

    return comment

def load_data_and_labels(file1, file2,file3,file4,file5,file6):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #start=time.time()
    file1_examples = readdata(file1)
    file2_examples = readdata(file2)
    file3_examples = readdata(file3)
    file4_examples = readdata(file4)
    file5_examples = readdata(file5)
    file6_examples = readdata(file6)

    x_text = file1_examples + file2_examples+file3_examples+file4_examples+file5_examples+file6_examples
    # end = time.time()
    #print(end-start)
    #exit(1)
    x_pre=[]

    for content in x_text:
        #content=remove_Stopword(tokenize(normalize_Text(content)))
        content.strip()
        x_pre.append(content)

    # Generate labels
    label1 = [[1,0,0,0,0,0] for _ in file1_examples]
    label2 = [[0,1,0,0,0,0] for _ in file2_examples]
    label3 = [[0,0,1,0,0,0] for _ in file3_examples]
    label4 = [[0,0,0,1,0,0] for _ in file4_examples]
    label5 = [[0,0,0,0,1,0] for _ in file5_examples]
    label6 = [[0,0,0,0,0,1] for _ in file6_examples]

    y = np.concatenate([label1, label2, label3, label4, label5, label6], 0)#np.concatenate: nối mảng.
    return [x_pre, y]#trả về


def batch_iter(data, batch_size, num_epochs, shuffle):#shuffle parameter:là có trộn dữ liệu hay không?
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1#epoch:tap du lieu
    # num_batches_per_epoch
    for epoch in range(num_epochs):#số lần đi qua tập dữ liệu
        # Shufle the data at each epoch,trộn data ở mỗi epoch
        if shuffle:#nếu có trộn(True)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            #np.random.permutation():đảo ngẫu nhiên các element trong mảng chỉ số của dữ liệu
            shuffled_data = data[shuffle_indices]#trộn dữ liệu theo index
        else:#ngược lại thì không trộn
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):#chạy các batch trong tập dữ liệu
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]#lấy dữ liệu ra của mỗi batch của tập dữ liệu
            # generator là một hàm trả kết quả về là một chuỗi kết quả thay vì một giá trị duy nhất.
            #Mỗi lần lệnh yield được chạy, nó sẽ sinh ra một giá trị mới. (Vì thế nó mới được gọi là generator)
#load_data_and_labels("./predata/pre1.txt","./predata/pre2.txt","./predata/pre3.txt","./predata/pre4.txt","./predata/pre5.txt","./predata/pre6.txt")
