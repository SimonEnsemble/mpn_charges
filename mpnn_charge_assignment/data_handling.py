
# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 20, 2019
# // PURPOSE     : Generate data_list to feed to dataloader
# // SPECIAL NOTES: It assumes that there are zzz.edge_info, zzz_node_feature.npy, and zzz_node_labels.npy in the "input" folder
# // ===============================
# // Change History: 1.0: initial code: wrote and tested.
# //
# //==================================

__author__ = "Ali Raza"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = ""
__version__ = "1.0"
__maintainer__ = "ali raza"
__email__ = "razaa@oregonstate.edu"
__status__ = "done"

import numpy as np
import torch
import os
from torch_geometric.data import Data, DataLoader


def data_handling(graphs_folder, crystal):
	data_list = []
	node_features = np.load(graphs_folder + "/" + crystal + "_node_features.npy")
	file1 = open(graphs_folder+"/" + crystal + ".edge_info", "r")
	line = file1.readline()
	lines = file1.readlines()
	edge_in = []
	edge_out = []
	distance = []
	for x in lines:
		edge_in.append(int(x.split(',')[0]))
		edge_out.append(int(x.split(',')[1]))
		# It is undirected graph. Message needs to flow in both directions
		edge_in.append(int(x.split(',')[1]))
		edge_out.append(int(x.split(',')[0]))
		distance.append(float(x.split(',')[2]))
	file1.close()
	edges = [edge_in, edge_out]
	edges = np.asarray(edges)
	x = torch.tensor(node_features, dtype=torch.double)
	edge_index = torch.tensor(edges, dtype=torch.long)
	data_list.append(Data(x=x, edge_index=edge_index))

	return data_list
