import pandas as pd
import numpy as np
import datetime as dt

class DataReader:
    def __init__(self, data):
        self.MovieDataFrame = pd.read_csv(data)
        self.weightsArray = []
        self.Output = []

    def classifyRevenue(self):
        for ind, row in self.MovieDataFrame.iterrows():
            rev = float(row["revenue"])
            if rev < 13612065.75:
                self.setNewAttribute(ind, "revenue", "Bust")
            elif rev > 13612065.75 and rev < 124296916.75:
                self.setNewAttribute(ind, "revenue", "Average")
            else:
                self.setNewAttribute(ind, "revenue", "Blockbuster")

    def formatArrays(self, phrase):
        for ind, row in self.MovieDataFrame.iterrows():
            arr = [c for c in str(row[phrase]).split("|")]
            self.setNewAttribute(ind, phrase, arr)

    def formatDates(self):
        DateDict = {2:[1,2,8], 1:[9,10], 3:[3,4], 5:[5,6,7], 4:[11,12]}
        for ind, row in self.MovieDataFrame.iterrows():
            DateObj = dt.datetime.strptime(str(row["release_date"]), '%m/%d/%Y')
            for key, val in DateDict.items():
                if int(DateObj.month) in val:
                    self.setNewAttribute(ind,"release_date", key)

    def formatVotes(self):
        self.MovieDataFrame["voteScore"] = 0
        for ind, row in self.MovieDataFrame.iterrows():
            self.setNewAttribute(ind, "voteScore", (int(row["vote_count"]) * int(row["vote_average"])))
        self.MovieDataFrame.drop(["vote_count", "vote_average"], axis = 1, inplace = True)
    
    def removeNA(self):
        self.MovieDataFrame.drop(["id", "production_companies", "director","tagline", "imdb_id", "original_title", "homepage", "keywords", "overview", "budget_adj", "revenue_adj"], axis =1, inplace = True)
        self.MovieDataFrame.dropna(subset = ["release_date", "cast"], inplace = True) 
        for ind, row in self.MovieDataFrame.iterrows():
            if (int(row["revenue"]) == 0 or int(row["budget"]) == 0):
                self.MovieDataFrame.drop(ind, inplace= True)

    def formatData(self):
        self.removeNA()
        self.formatDates()
        self.formatArrays("cast")
        self.formatArrays("genres")
        self.formatVotes()
        self.classifyRevenue()
        self.MovieDataFrame = self.MovieDataFrame[['popularity', 'budget', 'cast', 'release_date', 'voteScore', 'runtime', 'release_year', "genres", 'revenue']]
        

    def setNewAttribute(self, index, col, value):
        self.MovieDataFrame.at[index, col] = value
        return self.getMovieDF()

    def getOutput(self):
        return self.Output


    def getMovieDF(self):
        return self.MovieDataFrame
        

    def saveData(self):
        pass

    def getWeights(self):
        pass

    def setWeights(self):
        pass



if __name__ == "__main__":
    arr = DataReader("tmdb-movies.csv")
    arr.formatData()
    arr.getOutput()
    
    

    