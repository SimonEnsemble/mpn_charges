
# # Deployment
# ### Charge predictions on graphs

#  // ===============================
#  // AUTHOR     : Ali Raza (razaa@oregonstate.edu)
#  // CREATE DATE     : Feb 15, 2020
#  // PURPOSE     : charge predictions on graphs 
#  // SPECIAL NOTES: needs models.pt (contains models trained for gaussian and uniform correction)
#  // ===============================
#  // Change History: 0.1: initial code: wrote and tested.
#  // Change History: 0.2: updated code: used torch.long to increae accuracy 
#  // Change History:  0.3: Added argument parser
#  //
#  //==================================

# Libraries
from data_handling import *
from model import *
from torch_geometric.data import Data, DataLoader
import numpy as np
import argparse
import os
# ---------------------
# Parameters
# ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crit = torch.nn.L1Loss()

# for graphs decorations
hfont = {'fontname':'Times New Roman'}
fontsize_label_legend = 24

parser = argparse.ArgumentParser()
parser.add_argument("graphs_directory", help="string specifying the directory in ../building_graphs/ containing graphs")
parser.add_argument("graph_name", help="string of crystal name")
args = parser.parse_args()
deployment_graphs = args.graphs_directory
graph_name = args.graph_name
# print('\n>>> will search <{}> for graph <{}>'.format(deployment_graphs, graph_name))

# ---------------------
# Parameters    
# ---------------------
print("\treading graphs and generating data list...")
data_list = data_handling(deployment_graphs,graph_name) #, READ_LABELS = False)




# ---------------------
# Loading models
# ---------------------
# models = torch.load('./models_deployment.pt',map_location=device)
# model = models[1] # [0]: mean; [1]: gaussian
# torch.save(model.state_dict(), 'state_dict.pt')
# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#model = Net_gaussian_correction(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)
model = Net_gaussian_correction(74,10,4,30).to(device)
model.load_state_dict(torch.load('state_dict.pt',map_location=torch.device('cpu')))  # Choose whatever GPU device number you want
model.to(device)
model = model.double()
# print('is model running on cuda? : {}'.format(next(model.parameters()).is_cuda))
# ---------------------
# Predicting charges
# ---------------------
print('\tinferring charges...')
loader = DataLoader(data_list, batch_size=1)

with torch.no_grad():
    for data in loader:
        data = data.to(device)       
        data.x = data.x.type(torch.DoubleTensor).to(device)      
        features = data.x.to(device)        
        model.eval()
        pred, embedding, sigmas, uncorrected_mu = model(data)
        np.save("{}/{}_mpnn_charges".format(deployment_graphs, graph_name), pred.cpu().numpy())
        
print('\twritting charges to file...')

