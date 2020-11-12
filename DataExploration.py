import matplotlib.pyplot as plt
import numpy as np
from numpy import cov
import pandas as pd 
import seaborn as sns
import statistics
from scipy.stats import pearsonr

Data = pd.read_csv("DataRevExploration.csv", header = 0)


def plotScatter(x, y):
    m, b = np.polyfit(Data[x], Data[y], 1)
    plt.plot(Data[x], Data[y], 'o')
    plt.plot(Data[x], m*Data[x] + b)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(x + " vs " + y)
    plt.show()
    #plt.savefig("BlockBusterGraphs/"+ x + "vs" + y +".png")

#plotScatter(res[0],res[1])

"""
sns.boxplot(y=tf[1], x=tf[0], data=Data,palette="colorblind", showfliers = False)
plt.title("BoxPlot of Oscars and Revenues")
plt.xlabel("Revenu")
plt.ylabel("Oscars")
plt.show()
#plt.savefig("BoxPlotOvR.png")
"""



"""
covariance = cov(res[], Data["revenue"])
corr = pearsonr(Data["release_year"], Data["revenue"])

with open("CorelationStats", "a") as f:
    f.write("CORELATION BETWEEN " + "release_year" + " AND REVENUE: COVARIANCE: " + str(covariance) + "PEARSONS: " + str(corr) + "\n")
    """
