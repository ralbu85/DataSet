import matplotlib.pyplot as plt
import numpy as np

def graph(func, X,y):
    mean=[]
    std=[]
    alpha=np.logspace(-4, -0.5, 30)
    for i in alpha:
        temp=func(X,y,alpha=i)
        mean.append(temp[0])
        std.append(temp[1])
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)    
    plt.semilogx(alpha,mean)
    plt.xlabel('mean')
    plt.subplot(1,2,2)
    plt.semilogx(alpha,std)
    plt.xlabel('std')
    plt.show()
    return 1
