
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

def data_handling(graphs_folder):
	import os
	import numpy as np
	import torch
	from torch_geometric.data import Data, DataLoader

	# crystals_names = os.listdir("xtals/") # returns list
	# crystals_names = [s.split('.')[0] for s in crystals_names]

	crystals_names = os.listdir(graphs_folder + "/")  # returns list
	crystals_names.sort()
	#crystals_names = [s.split('.')[0] for s in crystals_names]  # remnove extensions
	crystals_names = [ s.replace(".npy","") for s in crystals_names]  # remnove extensions
	crystals_names = [ s.replace(".edge_info", "") for s in crystals_names]  # remnove extensions
	crystals_names = [s.split('_node_features')[0] for s in crystals_names]  # remove node_features
	crystals_names = [s.split('_node_labels')[0] for s in crystals_names]  # node node_labels

	crystals_names = list(dict.fromkeys(crystals_names))
	np.save("crystals_name", crystals_names)

	print("Number of crystals: ", len(crystals_names))

	data_list = []

	for crystal in crystals_names:

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

		x = torch.tensor(node_features, dtype=torch.float)
		# y = torch.tensor(labels, dtype=torch.float)
		edge_index = torch.tensor(edges, dtype=torch.long)
		#     print("X.size: {}, y.size: {}, edge_index.size: {}".format(x.size(), y.size(), edge_index.size()))
		# data_list.append(Data(x=x, y=y, edge_index=edge_index))
		data_list.append(Data(x=x, edge_index=edge_index))

	return data_list
