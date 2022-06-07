#import all the necessary libraries

import pandas as pd   #for data manipulation
import networkx as nx  #for drawing graphs
import matplotlib.pyplot as plt  #for drawing graphs

#for creating the Bayesian Network
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

#To get the data and derive a few new variables for usage in the model
pd.options.display.max_columns=10

#To read the Student Performance in Tests data csv
df=pd.read_csv('StudentsPerformance.csv', encoding='utf-8') # "Student Performance" data set is collected from Kaggle

# To drop records where target gender=NaN
df=df[pd.isnull(df['gender'])==False]


#Creating new ones for variables that I want to use in the probability model
df['MathScoreNew']=df['math score'].apply( lambda x: 'MathScore>80' if x>80 else
                                            'MathScore<=80')
df['WritingScoreNew']=df['writing score'].apply( lambda x: 'WritingScore>80' if x>80 else
                                            'WritingScore<=80')
df['ReadingScoreNew']=df['reading score'].apply( lambda x: 'ReadingScore>80' if x>80 else
                                            'ReadingScore<=80')
#df a snapshot of data

# This function helps to calculate probability distribution, which goes into the BN (can handle up 1 parent only)
def probability(data, child, parent=None):
    if parent==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent!=None:
                # Caclucate probabilities
           prob=pd.crosstab(data[parent],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob


# To create nodes that automatically calculate probabilities in the network
Gender = BbnNode(Variable(0, 'Gender', ['male', 'female']), probability(df, child='gender'))
MathScore = BbnNode(Variable(1, 'MathScore', ['MathScore>80', 'MathScore<=80']), probability(df, child='MathScoreNew', parent='gender'))
WritingScore = BbnNode(Variable(2, 'WritingScore', ['WritingScore>80', 'WritingScore<=80']), probability(df, child='WritingScoreNew' ,parent='gender'))
ReadingScore = BbnNode(Variable(3, 'ReadingScore', ['ReadingScore>80', 'ReadingScore<=80']), probability(df, child='ReadingScoreNew', parent='WritingScoreNew'))
#ReadingScore = BbnNode(Variable(3, 'ReadingScore', ['ReadingScore>80', 'ReadingScore<=80']), probability(df, child='ReadingScoreNew'))



# To Create the Configuration of the Network.
bbn = Bbn() \
    .add_node(WritingScore) \
    .add_node(ReadingScore) \
    .add_node(Gender) \
    .add_node(MathScore) \
    .add_edge(Edge(WritingScore, ReadingScore, EdgeType.DIRECTED)) \
    .add_edge(Edge(Gender, WritingScore, EdgeType.DIRECTED)) \
    .add_edge(Edge(Gender, MathScore, EdgeType.DIRECTED))


# Convert the BN to a join tree
join_tree = InferenceController.apply(bbn)


# Define a function for printing marginal probabilities (marginal probability is the probability of an event irrespective of the outcome of another variable(s))
def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')


# Use the above function to print marginal probabilities
print_probs()

# To add evidence of events that happened so probability distribution can be recalculated
def evidence(e, node, new, value):
    e = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name(node)) \
        .with_evidence(new, value) \
        .build()
    join_tree.set_observation(e)


# Use above function to add evidence
evidence('ev1', 'Gender', 'female', 1.0)
#evidence('ev1', 'Gender', 'male', 1.0)
#evidence('ev2', 'WritingScore', 'WritingScore>80', 1.0)
#evidence('ev3', 'MathScore', 'MathScore>80', 1.0)


# Print marginal probabilities after recalculation
print('Probability recalculated given evidence: ')
print_probs() #to calculate the probabilities given an evidence set.

# Drawing the tree graph. 
# Set node positions
pos = {0: (1, 2.5), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

# To set options for the outlook of the graph. 
options = {
    "font_size": 12,
    "node_size": 8000,
    "node_color": "Yellow",
    "edgecolors": "Black",
    "edge_color": "blue",
    "linewidths": 5,
    "width": 5, }

# To generate the graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()

#Questions raised: 1) How do you specify the network configuration?
# ans. I specified the network configuration from line 55 to 62. I configured it by testing out different columns from the dataset as the parent and putting them in evidence
# set to observe any changes to other probabilities.
# 2) How do you input arbitrary query like comparing Pr(ReadingScore|Gender) vs. Pr(ReadingScore)?
#ans. I would compare them by updating the probability (line: 47-50) and network configuration (line: 55-62). From those lines,
# I will remove other variables such as, WritingScore and MathScore. Update ReadingScore's parent to Gender. I will note the change to the probability. Then I will remove the parent, and
# observe its probaility as a single node to note its probability.

