
# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 22, 2019
# // PURPOSE     : main function.
# // SPECIAL NOTES: Uses data_handling.py and model.py
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


from data_handling import *
from model import *
from torch_geometric.data import Data, DataLoader
import os
import numpy as np
from sklearn.metrics import roc_auc_score

MAX_EPOCHS = 10
BATCH_SIZE = 32

# if not(os.path.exists("data_list.npy")):
#     data_list = data_handling()
#     np.save("data_list.npy", data_list)
print("----------------------------------------------")
print(">>> reading files and generating data_list")
data_list = data_handling()
print("...done")
print("----------------------------------------------")
print()

# dividing data into testing and training
NUM_NODE_FEATURES = data_list[0]['x'].shape[1]
data_size = len(data_list)
cut = int(data_size*0.7)
train_dataset = data_list[:cut]
test_dataset = data_list[cut:]

print("Training crystals: {}".format(len(train_dataset)) )
print("Testing crystals: {}".format(len(test_dataset)) )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



# initializing the model

device = torch.device('cuda')
model = Net(NUM_NODE_FEATURES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.MSELoss()



# module for training
def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)
	

# module for evaluating
def evaluate(loader):
    model.eval()

    predictions = []
    labels = []
    loss_all = 0

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            pred = model(data)
            label = data.y.to(device)
            loss = crit(pred, label)
            loss_all += data.num_graphs * loss.item()


            # predictions.append(pred)
            # labels.append(label)

    # predictions = np.hstack(predictions)
    # labels = np.hstack(labels)
    # print("predictions size: ", predictions.shape)
    # print("label size: ", label.shape)
    # todo
    # return roc_auc_score(labels, predictions)
    # return crit(labels, predictions)
    return loss_all/len(loader)
	
for epoch in range(MAX_EPOCHS):
    loss = train()
    train_acc = evaluate(train_loader)
    # val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    # print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
    #       format(epoch, loss, train_acc, val_acc, test_acc))
    print('Epoch: {:03d}, Loss: {:.5f}, train_loss: {:.5f}, test_loss: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))

