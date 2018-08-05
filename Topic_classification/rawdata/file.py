import json

with open("1.json","r") as g:
	datas=json.load(g)
	for data in datas :
	  print(data["content"])
