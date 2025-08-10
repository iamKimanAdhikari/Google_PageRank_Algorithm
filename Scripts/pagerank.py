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

        #Zip is used to aggregate the elements from the dataframe in a tuple
        for source, destination in zip(self.df["From"], self.df["To"]):
            #Source is the key and destination is included in a list
            self.graph[source].append(destination)

            #Adding source and destination to the nodes set to remove repitition
            self.nodes.add(source)
            self.nodes.add(destination)
    
    def calculate_pagerank(self):
        N = len(self.nodes) #counting the number of unique nodes
        self.node_to_index = {node: i for i,node in enumerate(self.nodes)}
        self.index_to_node = {i:node for node, i in self.node_to_index.items()} 
        
        #creating a list for the initial page rank of the nodes
        ranks = [1.0/N]*N
        #container to store the calculated rank of the nodes
        new_ranks = [0.0] * N

def main():
    PageRank()

if __name__  == "__main__":
    main()