
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

def data_handling(graphs_folder, READ_LABELS = True):
	import os
	import numpy as np
	import torch
	from torch_geometric.data import Data, DataLoader

	# crystals_names = os.listdir("xtals/") # returns list
	# crystals_names = [s.split('.')[0] for s in crystals_names]

	crystals_names = os.listdir(graphs_folder + "/")  # returns list
	crystals_names.sort()
	# crystals_names = [s.split('.')[0] for s in crystals_names]  # remnove extensions
	crystals_names = [s.replace(".npy", "") for s in crystals_names]  # remnove extensions
	crystals_names = [s.replace(".edge_info", "") for s in crystals_names]  # remnove extensions
	crystals_names = [s.split('_node_features')[0] for s in crystals_names]  # remove node_features
	crystals_names = [s.split('_node_labels')[0] for s in crystals_names]  # node node_labels

	crystals_names = list(dict.fromkeys(crystals_names))
	np.save("crystals_name", crystals_names)

	print("total crystals: ", len(crystals_names))

	data_list = []

	for crystal in crystals_names:

		node_features = np.load(graphs_folder + "/" + crystal + "_node_features.npy")
		if (READ_LABELS):
			labels = np.load(graphs_folder + "/" + crystal + "_node_labels.npy")
			#==========================================================================================
			# labels = labels[:, 0]  # for testing purposes. Labels contain the same data as features by mistake.
			# labels = np.random.rand(node_features.shape[0])
			# print("size of node_features ", node_features.shape)
			# print("size of labels: ", labels.shape)
			# print("size of labels2: ", labels2.shape)
			if (labels.shape[0] != node_features.shape[0]):
				print("something is wrong")
				print(crystal)
				print("size of node_features ", node_features.shape)
				print("size of labels: ", labels.shape)
				print(crystal)
				# exit()
				continue
			#==========================================================================================

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
		if (READ_LABELS):
			y = torch.tensor(labels, dtype=torch.double)
		edge_index = torch.tensor(edges, dtype=torch.long)
		#     print("X.size: {}, y.size: {}, edge_index.size: {}".format(x.size(), y.size(), edge_index.size()))
		if (READ_LABELS):
			data_list.append(Data(x=x, y=y, edge_index=edge_index))
		else:
			data_list.append(Data(x=x, edge_index=edge_index))

	return data_list
