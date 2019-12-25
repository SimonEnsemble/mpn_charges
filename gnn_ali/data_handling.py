
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

def data_handling():
	import os
	import numpy as np
	import torch
	from torch_geometric.data import Data, DataLoader

	# crystals_names = os.listdir("xtals/") # returns list
	# crystals_names = [s.split('.')[0] for s in crystals_names]

	crystals_names = os.listdir("input/")  # returns list
	crystals_names = [s.split('.')[0] for s in crystals_names]  # remnove extensions
	crystals_names = [s.split('_node_features')[0] for s in crystals_names]  # remove node_features
	crystals_names = [s.split('_node_labels')[0] for s in crystals_names]  # node node_labels

	crystals_names = list(dict.fromkeys(crystals_names))


	print("Number of crystals: ", len(crystals_names))

	data_list = []

	for crystal in crystals_names:

		node_features = np.load("input/" + crystal + "_node_features.npy")
		labels = np.load("input/" + crystal + "_node_labels.npy")
		#==========================================================================================
		labels = labels[:, 0]  # for testing purposes. Labels contain the same data as features by mistake.
		# labels = np.random.rand(node_features.shape[0])

		# print("size of labels: ", labels.shape)
		# print("size of labels2: ", labels2.shape)
		# exit()

		#==========================================================================================

		file1 = open("input/" + crystal + ".edge_info", "r")
		line = file1.readline()
		lines = file1.readlines()
		edge_in = []
		edge_out = []
		distance = []
		for x in lines:
			edge_in.append(int(x.split(',')[0]))
			edge_out.append(int(x.split(',')[1]))
			distance.append(float(x.split(',')[2]))

		file1.close()

		edges = [edge_in, edge_out]
		edges = np.asarray(edges)

		x = torch.tensor(node_features, dtype=torch.float)
		y = torch.tensor(labels, dtype=torch.float)
		edge_index = torch.tensor(edges, dtype=torch.long)
		#     print("X.size: {}, y.size: {}, edge_index.size: {}".format(x.size(), y.size(), edge_index.size()))
		data_list.append(Data(x=x, y=y, edge_index=edge_index))

	return data_list
