import matplotlib.pyplot as plt
import numpy as np
from numpy import cov
import pandas as pd 
import seaborn as sns
from scipy.stats import pearsonr

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
    plt.savefig("BlockBusterGraphs/"+ x + "vs" + y +".png")


#plotScatter("runtime","revenue")

"""
sns.boxplot(y='cast', x='revenue', data=Data,palette="colorblind", showfliers = False)
plt.title("BoxPlot of Oscars and Revenues")
plt.xlabel("Revenu")
plt.ylabel("Oscars")
plt.savefig("BoxPlotOvR.png")
"""



covariance = cov(Data["release_year"], Data["revenue"])
corr = pearsonr(Data["release_year"], Data["revenue"])

with open("CorelationStats", "a") as f:
    f.write("CORELATION BETWEEN " + "release_year" + " AND REVENUE: COVARIANCE: " + str(covariance) + "PEARSONS: " + str(corr) + "\n")
