import os
import torch
import torch.nn as nn
from torch.optim import *
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from dataLoader import *
from model import *
torch.cuda.empty_cache()

train_loader, valid_loader, test_loader = LoadDatasets()
model = MRIGCN(nfeat=512, nhid=256, nclass=2, dropout=0.2)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

if device=='cuda:0':
    model = model.cuda()

print(model)
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total No of Parameters {} \nTotal no of trainable parameters {}".format(total_params, total_trainable_params))

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
vloss_list = []
vaccuracy_list = []
best_val_loss = 100000
num_epochs = 15
# Cross Entropy Loss 
error = nn.CrossEntropyLoss()

# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start = datetime.now()
model_dir = os.path.join(os.getcwd(), 'models')
if os.path.exists(model_dir)==False:
    os.mkdir(model_dir)
model_path = os.path.join(model_dir, f'{start}best-model-parameters.pt')

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), torch.eye(2).index_select(dim=0, index=labels).cuda()
            
            
        model.train(True)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        #print(inputs.shape)
        outputs = model(inputs)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % len(train_loader) == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            running_vloss, running_count = 0, 0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(valid_loader):
                    vinputs, vlabels = vdata
                    if torch.cuda.is_available():
                        vinputs, vlabels = vinputs.cuda(), vlabels.cuda()  
                    voutputs = model(vinputs)
                    vloss = error(voutputs, vlabels)
                    running_vloss += vloss.data
                    running_count += 1
                    
                    # Get predictions from the maximum value
                    vpredicted = torch.max(voutputs.data, 1)[1]

                    # Total number of labels
                    total += len(vlabels)
                    correct += (vpredicted == vlabels).sum()

                vaccuracy = 100 * (correct / float(total))
                # store loss and iteration
                v_loss = running_vloss/float(running_count)
                vloss_list.append(v_loss)
                vaccuracy_list.append(vaccuracy)

            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            #accuracy_list.append(accuracy)
            
        #if count % 5 == 0:
            # Print Loss
            print('Epoch: {} Training Loss: {}  Validation Loss: {} and Validation Accuracy: {} %'.format(int(count/len(train_loader)), loss.data, v_loss, vaccuracy))
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), model_path)
                print("Best Model Updated!\n\n")
