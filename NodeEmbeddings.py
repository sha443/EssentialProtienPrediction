import joblib
from node2vec import Node2Vec
import networkx as nx
import csv

# Step 1: Load PPI data and create graph
G = nx.Graph()

ppi_file = 'data/ppi/biogrid.txt'
node_embeddings_file = "biogrid_node_embeddings_256.csv"

with open(ppi_file, 'r') as f:
    for line in f:
        protein1, protein2 = line.strip().split()
        G.add_edge(protein1, protein2)

with joblib.parallel_backend('threading'):
    # Step 2: Generate walks and learn embeddings with Node2Vec
    node2vec = Node2Vec(G, dimensions=256, walk_length=80,
                        num_walks=300, workers=16)
    model = node2vec.fit(window=10, min_count=3, batch_words=10)

 # Step 3: Get embeddings and protein names
embeddings = {}
for node in G.nodes():
    try:
        embeddings[node] = model.wv[node]
    except KeyError:
        print(f"Embedding not found for node {node}.")

# Step 4: Write embeddings to CSV file with protein names
with open(node_embeddings_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header row with protein names and dimensions
    header = ['Name'] + [f'Dim_{i+1}' for i in range(model.vector_size)]
    writer.writerow(header)

    # Write embeddings
    for protein, embed in embeddings.items():
        writer.writerow([protein] + list(embed))

print(f"Node embeddings with protein names written to {node_embeddings_file}")
