import numpy as np
import torch as ch
from Model_Implementations import Model_Tanh_Ensemble, predict
import scipy.linalg
import utils
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.nn.functional import relu


constants = utils.CIFAR10()
ds = constants.get_dataset()
train_loader, test_loader = ds.make_loaders(batch_size=16, workers=8, data_aug=False)
X_train, Y_train, X_test, Y_test= [], [], [], []
for (x, y) in train_loader:
    X_train.append(x)
    Y_train.append(y)
for (x, y) in test_loader:
    X_test.append(x)
    Y_test.append(y)
X_train, Y_train = ch.cat(X_train), ch.cat(Y_train)
X_test, Y_test   = ch.cat(X_test), ch.cat(Y_test)


#1. num_chunks refers to how many models comprise the ensemble (4 is used in the paper); code_length/num_chunks shoould be an integer
#2. output_activation is the function to apply to the logits
#   a. one can use anything which gives support to positive and negative values (since output code has +1/-1 elements); tanh or identity maps both work
#   b. in order to alleviate potential concerns of gradient masking with tanh, one can use identity as well
#3. M is the actual coding matrix (referred to in the paper as H).  Each row is a codeword
#   note that any random shuffle of a Hadmard matrix's rows or columns is still orthogonal

code_length = 32
num_codes = code_length

M = scipy.linalg.hadamard(code_length, dtype=np.float32)
# replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier
# this change still ensures all codewords have dot product <=0
# since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
M[np.arange(0, num_codes, 2), 0] = -1
np.random.shuffle(M)
idx = np.random.permutation(code_length)
M = M[0:num_codes, idx[0:code_length]]

# device = ch.device('cuda:0')
device = ch.device('cpu')

# Convert to Tensor
M = ch.from_numpy(M).to(device)

m5 = Model_Tanh_Ensemble(M=M, device=device)
m5.to(device)

class CustomCIFAR10(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return (self.X[index], ch.stack([y_[index] for y_ in self.Y]))

    def __len__(self):
        return self.len

#map categorical class labels (numbers) to encoded (e.g., one hot or ECOC) vectors
def encodeData(Y, M):
    Y_ = ch.zeros(Y.shape[0], M.shape[1])
    for k in np.arange(M.shape[1]):
        Y_[:, k] = M[Y, k]
    return Y_

def epoch(model, opt, train_loader, val_loader):
    total_loss, total_metric = 0., 0.
    
    # model.eval()
    model.train()

    # Training loop
    for X,y in train_loader:
        X, y = X.to(device), y.to(device)
        y = y.permute(1, 0, 2)
        yp = model(X)
        loss = ch.sum(ch.stack([ch.mean(relu(1.0 - y[i] * yp[i])) for i in np.arange(model.num_chunks)]))
            
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Compute weird error
        total_metric += ch.mean(((yp * y) > 0).float())
        total_loss += loss.item() * X.shape[0]

        print(total_metric, total_loss)


    # Evaluation loop
    return total_metric / len(loader.dataset), total_loss / len(loader.dataset)


Y_train = encodeData(Y_train, m5.M)
Y_test  = encodeData(Y_test, m5.M)

Y_train_list=[]
Y_test_list=[]

start = 0
for k in np.arange(m5.num_chunks):
    end = start + m5.M.shape[1] // m5.num_chunks
    Y_train_list += [Y_train[:,start:end]]
    Y_test_list += [Y_test[:,start:end]]
    start=end

train_dataset = CustomCIFAR10(X_train, Y_train_list)
val_dataset   = CustomCIFAR10(X_test, Y_test_list)

train_loader = ch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
val_loader   = ch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

opt = ch.optim.Adam(m5.parameters(), lr=2e-4, eps=1e-7)
epoch(m5, opt, train_loader, val_loader)

# self.model.fit(X_train, Y_train_list, epochs=400, batch_size=200, shuffle=True, validation_data=[X_test, Y_test_list])
