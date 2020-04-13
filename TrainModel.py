import numpy as np
import torch as ch
from Model_Implementations import Model_Tanh_Ensemble, predict
import scipy.linalg
import utils
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.nn.functional import relu


#1. num_chunks refers to how many models comprise the ensemble (4 is used in the paper); code_length/num_chunks shoould be an integer
#2. output_activation is the function to apply to the logits
#   a. one can use anything which gives support to positive and negative values (since output code has +1/-1 elements); tanh or identity maps both work
#   b. in order to alleviate potential concerns of gradient masking with tanh, one can use identity as well
#3. M is the actual coding matrix (referred to in the paper as H).  Each row is a codeword
#   note that any random shuffle of a Hadmard matrix's rows or columns is still orthogonal


# Define device for all operations
device = ch.device('cuda:0')


class CustomCIFAR10(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return (self.X[index], ch.stack([y_[index] for y_ in self.Y]))

    def __len__(self):
        return self.len


def get_all_cifar10_data():
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
    return (X_train, Y_train), (X_test, Y_test)


def get_model(code_length = 32):
    num_codes = code_length

    M = scipy.linalg.hadamard(code_length, dtype=np.float32)
    # replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier
    # this change still ensures all codewords have dot product <=0
    # since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
    M[np.arange(0, num_codes, 2), 0] = -1
    np.random.shuffle(M)
    idx = np.random.permutation(code_length)
    M = M[0:num_codes, idx[0:code_length]]

    # Convert to Tensor
    M = ch.from_numpy(M).to(device)

    model = Model_Tanh_Ensemble(M=M, device=device)
    model.to(device)
    return model

# Map categorical class labels (numbers) to encoded (e.g., one hot or ECOC) vectors
def encodeData(Y, M):
    Y_ = ch.zeros(Y.shape[0], M.shape[1])
    for k in np.arange(M.shape[1]):
        Y_[:, k] = M[Y, k]
    return Y_


def epoch(model, opt, data_loader, pbar_stats=False):
    total_loss, total_metric = 0., 0.
    so_far = 0
    # Training loop
    iterator = data_loader
    if pbar_stats:
        iterator = tqdm(iterator)
    for X, y in iterator:
        X, y = X.to(device), y.to(device)
        y = y.permute(1, 0, 2)
        yp = model(X)
        loss = ch.sum(ch.stack([ch.mean(relu(1.0 - y[i] * yp[i])) for i in np.arange(model.num_chunks)]))
        
        # Training mode
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        so_far += X.shape[0]

        # Compute weird error
        total_metric += ch.sum(((yp * y) > 0).float()).cpu()
        total_loss += loss.cpu().item() * X.shape[0]

        if pbar_stats:
            iterator.set_description('Loss : %.4f Metric : %.4f' % (total_loss / so_far, total_metric / so_far))

    # Evaluation loop
    return total_metric / len(data_loader.dataset), total_loss / len(data_loader.dataset)


def train_model(model, loaders, epochs, save_every):
    opt = ch.optim.Adam(model.parameters(), lr=2e-4, eps=1e-7)
    train_loader, val_loader = loaders
    for e in range(epochs):
        # Train for epoch
        model.train()
        print("Training epoch %d" % (e + 1))
        _ = epoch(model, opt, train_loader, True)
        # Run validation
        model.eval()
        with ch.no_grad():
            m, l = epoch(model, None, val_loader)
        print("Validation epoch %d : metric %.4f loss %.4f" % (e + 1, m, l))
        # Save model
        if (e + 1) % save_every == 0:
            ch.save(model.state_dict(), 'models/checkpoint_%d.pth' % (e+1))
            print("Checkpoint saved")


def get_data_loaders(model):
    (X_train, Y_train), (X_test, Y_test) = get_all_cifar10_data()
    Y_train = encodeData(Y_train, model.M)
    Y_test  = encodeData(Y_test,  model.M)

    Y_train_list= []
    Y_test_list = []

    start = 0
    for k in np.arange(model.num_chunks):
        end = start + model.M.shape[1] // model.num_chunks
        Y_train_list += [Y_train[:,start:end]]
        Y_test_list += [Y_test[:,start:end]]
        start=end

    train_dataset = CustomCIFAR10(X_train, Y_train_list)
    val_dataset   = CustomCIFAR10(X_test, Y_test_list)

    train_loader = ch.utils.data.DataLoader(train_dataset, batch_size=180, shuffle=True,  num_workers=4)
    val_loader   = ch.utils.data.DataLoader(val_dataset,   batch_size=100, shuffle=False, num_workers=4)

    return (train_loader, val_loader)


if __name__ == "__main__":
    model = get_model()
    data_loaders = get_data_loaders(model)
    train_model(model, data_loaders, epochs=500, save_every=50)
