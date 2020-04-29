import torch as ch
from Model_Implementations import Model_Tanh_Ensemble, predict
from cleverhans.future.torch.attacks import projected_gradient_descent
import scipy.linalg
import numpy as np
from tqdm import tqdm
import utils


# Load M matrix
M = ch.load('models/M.pt').cuda()

# Load model
model = Model_Tanh_Ensemble(M=M)
model = ch.nn.DataParallel(model).cuda()
model.load_state_dict(ch.load('models/checkpoint_300.pth'))

# Switch to eval model
model.eval()

# Run normal evaluation
constants = utils.CIFAR10()
ds = constants.get_dataset()
num_examples = 0
acc = 0.0
_, test_loader = ds.make_loaders(batch_size=32, workers=8, data_aug=False, only_val=True)
iterator = tqdm(test_loader)
for im, label in iterator:
	im    = im.cuda()
	label = label.cuda()

	# Get model predictions
	y_p   = predict(model, im)
	preds = ch.argmax(y_p, 1)
	num_examples += label.shape[0]
	acc += (ch.sum(preds == label)).float()
	iterator.set_description('Accuracy : %.3f' % (100 * acc / num_examples))


class ECOCToNormal:
	def __init__(self, m):
		self.m = m

	def __call__(self, x):
		logits = predict(self.m, x)
		return logits

wrapped_model = ECOCToNormal(model)
# eps = 8/255
# norm = np.inf
eps = 0.5
norm = 2
nb_steps = 20
eps_iter = 2.5 * eps / nb_steps
misclass, total = 0, 0

_, test_loader = ds.make_loaders(batch_size=32, workers=8, data_aug=False, only_val=True)
iterator = tqdm(test_loader)
# Run adversarial evaluation
for im, label in iterator:
	# Run PGD attack
	advs = projected_gradient_descent(wrapped_model, im.cuda(), eps, eps_iter, nb_steps, norm, 0, 1)
	pert_labels = ch.argmax(wrapped_model(advs), 1).cpu()

	misclass += (pert_labels != label).sum().item()
	total    += len(label)

	iterator.set_description('Attack success rate : %f' % (100 * misclass / total))
