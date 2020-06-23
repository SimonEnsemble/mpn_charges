
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
parser.add_argument("graphs", help="string specifying the directory in ../building_graphs/ containing graphs")
args = parser.parse_args()
deployment_graphs = args.graphs
print('\n>>> will search <{}> for graphs'.format(deployment_graphs))

# ---------------------
# Parameters
# ---------------------
print(">>> reading files and generating data_list")
data_list = data_handling(deployment_graphs) #, READ_LABELS = False)
print("...done")
print()
print("Total MOFs: {} ".format(len(data_list)))


# ---------------------
# Loading models
# ---------------------
models = torch.load('./models_deployment.pt',map_location=device)
model = models[1] # [0]: mean; [1]: gaussian
print('is model running on cuda? : {}'.format(next(model.parameters()).is_cuda))
# ---------------------
# Predicting charges
# ---------------------
print('>>> predicting charges')
loader = DataLoader(data_list, batch_size=1)
crystals_names = np.load("crystals_name.npy")

with torch.no_grad():
    index_mof = 0
    for data in loader:
        data = data.to(device)       
        data.x = data.x.type(torch.DoubleTensor).to(device)      
        features = data.x.to(device)        
        model.eval()
        pred, embedding, sigmas, uncorrected_mu = model(data)
        np.save("{}/{}_mpnn_charges".format(deployment_graphs, crystals_names[index_mof]), pred.cpu().numpy())
        index_mof += 1
        if index_mof % 10 == 0:
            print('||| Done with MOFs: {}'.format(index_mof), end="\r", flush=True)
print('||| Done with MOFs: {}'.format(index_mof), end="\r", flush=True)
print('||| Done with MOFs')
print('results are in <{}/>'.format(deployment_graphs))


# deleting temp files
os.remove("crystals_name.npy")