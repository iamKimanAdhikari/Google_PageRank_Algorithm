import pandas as pd
from collections import defaultdict
from pathlib import Path

class PageRank:
    def __init__(self, max_edges = None):
        self.filepath = Path(__file__).parent.parent/ "Dataset"
        self.file = self.filepath / "web-Google.txt" 
        self.max_edges = max_edges
    
    def load_graph(self):
        #Only checking the first few nodes for testing in the initial state
        self.df = pd.read_csv(
            self.file, 
            sep="\t",       #Not a csv file and the columns are separated by tab. 
            comment="#",    #There are a few lines as the description of the dataset
            names=["From", "To"],     #The names for the columns for the dataset
            nrows= self.max_edges     #limiting the number of rows in the dataframe
        )

        #Arranging the From and To nodes in a dictionary
        self.graph = defaultdict(list)

        #To ensure that the nodes aren't repeated
        self.nodes = set()
        

def main():
    PageRank()

if __name__  == "__main__":
    main()