import torch as ch
from Model_Implementations import Model_Tanh_Ensemble, predict
import scipy.linalg
import numpy as np
from tqdm import tqdm
import utils


# Load M matrix
M = ch.load('models/M.pt').cuda()

# Load model
model = Model_Tanh_Ensemble(M=M)
model = ch.nn.DataParallel(model).cuda()
model.load_state_dict(ch.load('models/checkpoint_25.pth'))

# Switch to eval model
model.eval()

# Run normal evaluation
constants = utils.CIFAR10()
ds = constants.get_dataset()
num_examples = 0
acc = 0.0
_, test_loader = ds.make_loaders(batch_size=128, workers=8, data_aug=False, only_val=True)
iterator = tqdm(test_loader)
for im, label in iterator:
	im    = im.cuda()
	label = label.cuda()

	# Encode labels
	y_p   = predict(model, im)
	preds = ch.argmax(y_p, 1)
	num_examples += label.shape[0]
	acc += (ch.sum(preds == label)).float()
	iterator.set_description('Accuracy : %.3f' % (100 * acc / num_examples))
