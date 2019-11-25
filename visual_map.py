from maps import map4,map8,map16,map32,map64,map128
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

bob = []
for row in map4: 
        for i in row: 
            if i =='S': 
                bob.append(0) 
            elif i =='H': 
                bob.append(1) 
            elif i =='F': 
                bob.append(2) 
            else: 
                bob.append(3)


bob = np.array(bob)
bob3 =bob.reshape(4,4)
plt.close()


sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/map4.png')
bob = []
for row in map8:
        for i in row: 
            if i =='S': 
                bob.append(0) 
            elif i =='H': 
                bob.append(1) 
            elif i =='F': 
                bob.append(2) 
            else: 
                bob.append(3) 
                
                
bob = np.array(bob)
bob3 =bob.reshape(8,8)



sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/map8.png')
plt.close()
bob = []
for row in map16:
        for i in row:
            if i =='S':
                bob.append(0)
            elif i =='H':
                bob.append(1)
            elif i =='F':
                bob.append(2)
            else:
                bob.append(3)


bob = np.array(bob)
bob3 =bob.reshape(16,16)



sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/map16.png')
plt.close()

bob = []
for row in map32:
        for i in row:
            if i =='S':
                bob.append(0)
            elif i =='H':
                bob.append(1)
            elif i =='F':
                bob.append(2)
            else:
                bob.append(3)


bob = np.array(bob)
bob3 =bob.reshape(32,32)



sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/map32.png')
plt.close()

bob = []
for row in map64:
        for i in row:
            if i =='S':
                bob.append(0)
            elif i =='H':
                bob.append(1)
            elif i =='F':
                bob.append(2)
            else:
                bob.append(3)


bob = np.array(bob)
bob3 =bob.reshape(64,64)



sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/map64.png')
plt.close()

bob = []
for row in map128:
        for i in row:
            if i =='S':
                bob.append(0)
            elif i =='H':
                bob.append(1)
            elif i =='F':
                bob.append(2)
            else:
                bob.append(3)


bob = np.array(bob)
bob3 =bob.reshape(128,128)



sns.heatmap(bob3,  cmap="YlGnBu", annot=False, cbar=False)
plt.savefig('graphs/map128.png')
plt.close()


