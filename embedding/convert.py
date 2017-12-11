

with open('em.txt','r') as f:
    data = f.readlines()

embeddings = []
words = []

for l in data:
    tmp = l.strip().split()
    words.append(tmp[0])
    embeddings.append(" ".join(tmp[1:]))

with open('embeddings.txt','w') as f:
    for e in embeddings:
        f.write(e+'\n')

with open('words.lst','w') as f:
    for w in words:
        f.write(w+'\n')
