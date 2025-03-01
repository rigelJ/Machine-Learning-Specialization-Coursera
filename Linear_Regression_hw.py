import numpy as np
import matplotlib.pyplot as plt_

m=x_train.shape[0]
predicted=np.dot(x_train,w)+b 

plt.scatter(x_train,y_train,marker='o',c='r')
plt.plot(x_train,predicted,c='b')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
"""


