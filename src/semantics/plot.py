import matplotlib.pyplot as plt
import numpy
file = open('data.txt','r')
data = file.readlines()
file.close()
count = 1.0
X = []
y =[]
count = 1
sum = 0
for x in data:
    sum+=float(x)
    if(count%10==0):
       y.append(sum)
       X.append(count*100)
       sum = 0
    count +=1
_, ax = plt.subplots()
ax.set_ylabel('Average Semantic Measure')
ax.set_xlabel('Iterations Trained')
ax.set_title('Performance Measure')
plt.plot(X,y)
plt.savefig('performance.svg',format='svg', dpi=1000)


