import matplotlib.pyplot as plt
import numpy as np
from numpy import cov
import pandas as pd 
import seaborn as sns
from scipy.stats import pearsonr

Data = pd.read_csv("CleanedDataRevenue.csv", header = 0)


AvgRevcat = [156433869, 218291850, 163619730, 97741254, 218234134, 84015521, 87412920, 74369825, 203153546, 257121500, 97349488, 91045775, 85748008, 106032069.99159664, 81405236, 83896935, 57287803, 24333724, 11115957, 42000000]
for i in AvgRevcat:
    f = [x for x in str(i)]
    print(len(f))
Cat = ['Action', 'Adventure', 'Science Fiction', 'Thriller', 'Fantasy', 'Crime', 'Western', 'Drama', 'Family', 'Animation', 'Comedy', 'Mystery', 'Romance', 'War', 'History', 'Music', 'Horror', 'Documentary', 'Foreign', 'TV Movie']
togeth = zip(AvgRevcat, Cat)
tf = sorted(togeth, key = lambda t: t[1])
res = list(zip(*tf)) 

      
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

plt.plot(res[1], res[0], 'o')
plt.show()
#plotScatter(res[0],res[1])


sns.boxplot(y=tf[1], x=tf[0], data=Data,palette="colorblind", showfliers = False)
plt.title("BoxPlot of Oscars and Revenues")
plt.xlabel("Revenu")
plt.ylabel("Oscars")
plt.show()
#plt.savefig("BoxPlotOvR.png")



"""
covariance = cov(res[], Data["revenue"])
corr = pearsonr(Data["release_year"], Data["revenue"])

with open("CorelationStats", "a") as f:
    f.write("CORELATION BETWEEN " + "release_year" + " AND REVENUE: COVARIANCE: " + str(covariance) + "PEARSONS: " + str(corr) + "\n")
    """
