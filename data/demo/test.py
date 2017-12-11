import json
res=0
with open('train_tag.json') as f:
	for _ in f.readlines():
		info=json.loads(_)
		res=max(res,len(info['tags']))
		# print(len(info['tags']))

print("=============")
print(res)