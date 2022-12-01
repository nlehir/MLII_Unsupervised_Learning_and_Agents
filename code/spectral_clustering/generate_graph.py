"""
Generating a random graph
"""

import random

import numpy as np
from graphviz import Graph

dot = Graph(comment="Graph used to study Spectral Clustering")

colors = ["black", "aquamarine4", "goldenrod3", "darkorchid4"]

dot.edge("1", "2", color="aquamarine4")
dot.edge("0", "2", color="aquamarine4")
dot.edge("0", "1", color="aquamarine4")
dot.edge("1", "3", color="aquamarine4")
dot.edge("1", "5", color="aquamarine4")
dot.edge("2", "5", color="aquamarine4")
dot.edge("4", "3", color="aquamarine4")
dot.edge("4", "2", color="aquamarine4")

dot.edge("6", "7", color="aquamarine4")
dot.edge("6", "8", color="aquamarine4")
dot.edge("8", "7", color="aquamarine4")

dot.edge("11", "13", color="aquamarine4")
dot.edge("11", "12", color="aquamarine4")
dot.edge("11", "10", color="aquamarine4")
dot.edge("10", "12", color="aquamarine4")
dot.edge("12", "13", color="aquamarine4")
dot.edge("12", "9", color="aquamarine4")
dot.edge("13", "9", color="aquamarine4")

# visualize the graph
graph_name = "images/graph_to_cluster"
dot.render(graph_name)
