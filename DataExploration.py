import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns

Data = pd.read_csv("CleanedDataRevenue.csv", header = 0)




#Data.boxplot(column=["revenue"])
#plt.show()


def plotScatter(x, y):
    m, b = np.polyfit(Data[x], Data[y], 1)
    plt.plot(Data[x], Data[y], 'o')
    plt.plot(Data[x], m*Data[x] + b)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + " vs " + y)
    plt.show()
    #plt.savefig("BlockBusterGraphs/"+ x + "vs" + y +".png")


plotScatter("cast","revenue")
"""
sns.boxplot(y='revenue', x='release_date', data=Data,palette="colorblind", showfliers = False)
plt.title("BoxPlot of Release Dates and Revenues")
plt.xlabel("release date")
plt.ylabel("revenue")
plt.savefig("BoxPlotRDvR.png")
"""