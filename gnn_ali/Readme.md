###### Base line for predicting charge. 
- `data_handling.py` -- assumes that there are `zzz.edge_info`, `zzz_node_feature.npy`, and `zzz_node_labels.npy` in the `input` folder, where zzz is the name of the molecule. (would have been better if they were in one file instead of different files for different molecules), and generates data_list. 
- `model.py` -- contains the network and GNN layer (right now it is `Gated Graph Conv 2016`). I am not using edge distance information right now. There are 4 layers of GNN (T=4) and final embedding size is 100. Then there is a linear layer to predict label (charge for each node). 
- `main.py` -- main file. Includes module for training and testing . Get the data_list from data_handling and load it to the DataLoader API for (batching processing). It uses 70% for training and 30% for testing. It is using MSE loss and Adam optimizer

