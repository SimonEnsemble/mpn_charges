# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 22, 2019
# // PURPOSE     : main function.
# // SPECIAL NOTES: Uses data_handling.py and model.py
# // ===============================
# // Change History: 1.0: initial code: wrote and tested.
# // Change History: 1.5: Added: minibatches, training/testing/validation split
# // Change History: 1.6: Troubleshooting: fixed bugs related to incorrect handling of minibatches
# // Change History: 2.0: Chagnes: Interface for wrapper for testing and evaluation
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


from model import *
from torch_geometric.data import Data, DataLoader
import numpy as np
from matplotlib import pyplot as plt
import math
import copy



def charge_prediction_system(train_loader, valid_loader,NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE,train_data_size, valid_data_size, MAX_EPOCHS, iteration, system, crit = torch.nn.L1Loss()):
	# initializing the model

	device = torch.device('cuda')
	if (system == 'vanilla' or system == 'soft_con'):
		print(">>> vanilla model")
		model = Net_vanilla(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)
	elif(system == 'mean_cor'):
		print(">>> mean_correction_model")
		model = Net_mean_correction(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)
	else:
		print(">>> gaussian_correction_model")
		model = Net_gaussian_correction(NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
	model = model.double()

	train_total_loss = []
	valid_total_loss = []
	min_valid_loss = float("inf")
	rebound = 0 # number of epochs that validation loss is increasing
	for epoch in range(MAX_EPOCHS):
		model.train()
		loss_all = 0
		for data in train_loader:
			data = data.to(device)
			optimizer.zero_grad()
			pred, _ , _, _= model(data)

			label = data.y.to(device)
			if (system == 'soft_con'):
				# print("soft constraint loss")
				loss = 100*crit(pred, label) + abs(sum(pred))
			else:
				loss = crit(pred, label)
			loss.backward()
			loss_all += data.num_graphs * loss.item()
			optimizer.step()
# 			print("||| PREDICTION SUM FOR ONE MOF: ", torch.sum(pred ))
		loss_epoch = loss_all / train_data_size

		# evaluating model
		model.eval()
		loss_all = 0
		with torch.no_grad():
			for data in train_loader:
				data = data.to(device)
				pred, _, _,_  = model(data)
				label = data.y.to(device)
				if (system == 'soft_con'):
					loss = 100 * crit(pred, label) + abs(sum(pred))
				else:
					loss = crit(pred, label)
				loss_all += data.num_graphs * loss.item()
		train_acc = loss_all / train_data_size
		train_total_loss.append(train_acc)
		# evaluating valid dataset
		model.eval()
		loss_all = 0
		with torch.no_grad():
			for data in valid_loader:
				data = data.to(device)
				pred, _, _, _ = model(data)
				label = data.y.to(device)
				if (system == 'soft_con'):
					loss = 100 * crit(pred, label) + abs(sum(pred))
				else:
					loss = crit(pred, label)
				loss_all += data.num_graphs * loss.item()
		valid_acc = loss_all / valid_data_size

		valid_total_loss.append(valid_acc)
		if valid_acc <= min_valid_loss: # keep tracking of model with lowest validation loss
			torch.save(model.state_dict(), './results/loss_iteration_' + str(iteration)+'_system_' + system + '.pth')
			min_valid_loss = valid_acc
			model_min = copy.deepcopy(model)
			rebound = 0
		else:
			rebound += 1
		if(epoch%10==0):
			print('Epoch: {:03d}, Loss: {:.5f}, train_loss: {:.5f}, valid_loss: {:.5f}'.
			  format(epoch+1, loss_epoch, train_acc, valid_acc))

		if rebound > 100: # early stopping criterion
			break
# 			pass

	hfont = {'fontname':'DejaVu Sans'}
	fontsize_label_legend = 24
	plt.figure(figsize=(8,8), dpi= 80)
	plt.plot(train_total_loss, label="train loss", color='dodgerblue', linewidth = 1.5)
	plt.plot(valid_total_loss, label="valid loss", color='red', linewidth = 1.5)
	plt.legend(frameon=False, prop={'size': 22})
	plt.xlabel('Epochs', fontsize=fontsize_label_legend, **hfont)
	plt.ylabel('Loss', fontsize=fontsize_label_legend, **hfont)
	plt.legend(frameon=False, prop={"family":"DejaVu Sans", 'size': fontsize_label_legend})
	plt.tick_params(axis='both', which='major', labelsize=17)
	plt.savefig('./results/loss_iteration_' + str(iteration)+'_system_' + system+'.png')
	plt.show()

	return model_min
