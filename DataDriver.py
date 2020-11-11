from DataReader import DataReader
import pandas as pd


class DataDriver:
    def __init__(self, oscars):
        self.Data = DataReader("tmdb-movies.csv")
        self.Data.formatData()
        self.OscarFile = pd.read_csv(oscars)
        self.ActorsDictionary = {}
        self.MovieDF = self.Data.getMovieDF()
        self.Categories = ["ACTOR", "ACTRESS", "ACTOR IN A SUPPORTING ROLE", "ACTRESS IN A SUPPORTING ROLE", "ACTOR IN A LEADING ROLE", "ACTRESS IN A LEADING ROLE"]
        self.OutputData = self.Data.getOutput()
        self.cleanOscarData()

    def scoreGenres(self):
        genreList = ['Action', 'Adventure', 'Science Fiction', 'Thriller', 'Fantasy', 'Crime', 'Western', 'Drama', 'Family', 'Animation', 'Comedy', 'Mystery', 'Romance', 'War', 'History', 'Music', 'Horror', 'Documentary', 'Foreign', 'TV Movie']
        GenreScore = {k:v for (k,v) in zip(genreList,list(reversed(range(len(genreList) + 1))))}
        for ind, row in self.MovieDF.iterrows():
            score = 1
            for genre in row["genres"]:
                score *= GenreScore[genre]
            self.Data.setNewAttribute(ind, "genres", score)

    def setActorsDict(self):
        for ind, row in self.MovieDF.iterrows():
            for actor in row["cast"]:
                    self.ActorsDictionary[actor] = 0
        self.scoreOscars()

    def cleanOscarData(self):
        self.OscarFile.drop(["year"], axis = 1, inplace = True)
        for ind, row in self.OscarFile.iterrows():
            if row["category"] not in self.Categories:
                self.OscarFile.drop([ind], inplace = True)
        self.setActorsDict()

    def scoreOscars(self):
        for ind, row in self.OscarFile.iterrows():
            if row["winner"]:
                if row["entity"] in self.ActorsDictionary.keys():
                    val = self.ActorsDictionary[row["entity"]]
                    self.ActorsDictionary[row["entity"]] = val + 1
            elif row["entity"] in self.ActorsDictionary.keys():
                val = self.ActorsDictionary[row["entity"]]
                self.ActorsDictionary[row["entity"]] = val
        self.AddScores()

    def SearchDict(self, dic, name):
        for key, val in dic.items():
            if key == name:
                return val
        return 0

    def IterateScore(self,dic, arr):
        Score = 0
        for person in arr:
            Score += self.SearchDict(dic, person)
        return Score

    def AddScores(self):
        for ind, row in self.MovieDF.iterrows():
            self.MovieDF = self.Data.setNewAttribute(ind, "cast", self.IterateScore(self.ActorsDictionary, row["cast"]))
    
    def setRevOutput(self):
        df1 = pd.get_dummies(self.MovieDF["revenue"])
        self.MovieDF = pd.concat([self.MovieDF, df1], axis = 1)
        self.MovieDF.drop(["revenue"], axis = 1, inplace = True)


    def SaveData(self):
        self.MovieDF.to_csv("NNTest.csv", index=False)

if __name__ == "__main__":
    ActorS = DataDriver("data_csv.csv")
    ActorS.scoreGenres()
    ActorS.setRevOutput()
    ActorS.SaveData()
