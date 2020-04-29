import numpy as np
import torch as ch
import torch.nn as nn
from torch.nn.functional import relu
import kornia


class ClassBlender(nn.Module):
    def __init__(self,  attenuation):
        super(ClassBlender, self).__init__()
        self.attenuation = attenuation

    def forward(self, x):
        if self.training:
            x_permuted = x[ch.randperm(x.shape[0])]
            angles = ((180 * ( 2 * ch.rand(x.shape[0]) - 1))).to(x.get_device())
            shifts = (4 * (2 * ch.rand(x.shape[0], 2) -1)).to(x.get_device())
            inputs_permuted_translated = kornia.translate(x_permuted, shifts)
            x_adjusted = kornia.rotate(inputs_permuted_translated, angles)
            x_adjusted = ch.clamp(x_adjusted, 0, 1)
                        
            return (1.0 - self.attenuation) * x + self.attenuation * x_adjusted
        return x

class DataAugmenter(nn.Module):
    def __init__(self):
        super(DataAugmenter, self).__init__()

    def forward(self, x):
        if self.training:
            angles = ((15 * (2 * ch.rand(x.shape[0]) - 1))).to(x.get_device())
            shifts = (4 * (2 * ch.rand(x.shape[0], 2) - 1)).to(x.get_device())
            inputs_shifted = kornia.translate(x, shifts)
            inputs_shifted_rotated = kornia.rotate(inputs_shifted, angles)
            condition = (ch.rand([x.shape[0], 1, 1, 1]) < 0.5).to(x.get_device())
            inputs_shifted_rotated_flipped = condition * kornia.hflip(inputs_shifted_rotated) + (~condition) * inputs_shifted_rotated          
            return inputs_shifted_rotated_flipped 
        return x


class Model_Tanh_Ensemble(nn.Module):    
    def __init__(self, M, noise_stddev=0.032, blend_factor=0.032, num_chunks=4):
        super(Model_Tanh_Ensemble, self).__init__()
        self.num_chunks = num_chunks
        self.M =  M
        self.noise_stddev = noise_stddev
        self.blend_factor = blend_factor
        self.n = self.M.shape[1] // self.num_chunks

        self.conlayer1, self.conlayer2, self.conlayer3 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.conlayer4, self.conlayer5 = nn.ModuleList(), nn.ModuleList()
        self.dense1, self.dense2 = nn.ModuleList(), nn.ModuleList()

        for k in np.arange(0, self.num_chunks):

            self.conlayer1.append(nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),
                nn.ELU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.ELU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=17),
                nn.ELU(),
                nn.BatchNorm2d(32)))

            self.conlayer2.append(nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=17),
                nn.ELU(),
                nn.BatchNorm2d(64)))

            self.conlayer3.append(nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=17),
                nn.ELU()))

            conlayer4, conlayer5 = nn.ModuleList(), nn.ModuleList()
            dense1, dense2 = nn.ModuleList(), nn.ModuleList()
            for i in range(self.n):
                conlayer4.append(nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=18),
                    nn.ELU(),
                    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=17),
                    nn.ELU(),
                    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=17),
                    nn.ELU()))

                # Pad bottom right for asymmetric case (just like TF)
                conlayer5.append(nn.Sequential(
                    nn.ConstantPad2d((0, 1, 0, 1), 0.0),
                    nn.Conv2d(144, 16, kernel_size=2),
                    nn.ELU(),
                    nn.ConstantPad2d((0, 1, 0, 1), 0.0),
                    nn.Conv2d(16, 16, kernel_size=2),
                    nn.ELU()))

                dense1.append(nn.Sequential(
                    nn.Linear(16384, 16),
                    nn.ELU(),
                    nn.Linear(16, 8),
                    nn.ELU(),
                    nn.Linear(8, 4),
                    nn.ELU(),
                    nn.Linear(4, 2)))

                dense2.append(nn.Linear(2, 1))

            self.conlayer4.append(conlayer4)
            self.conlayer5.append(conlayer5)
            self.dense1.append(dense1)
            self.dense2.append(dense2)


    def forward(self, inp):
        outputs      = []
        penultimate  = []
        penultimate2 = []
        
        for k in np.arange(0, self.num_chunks):
            x = inp
            # Convert to grayscale
            rgb_weights = [0.2989, 0.5870, 0.1140]
            x_gs = x[:,0] * rgb_weights[0] + x[:,1] * rgb_weights[1] + x[:,2] * rgb_weights[2]
            x_gs = x_gs.unsqueeze(1)
           
            if self.training:
                x    += ch.randn(x.size()).to(x.get_device()) * self.noise_stddev
                x_gs += ch.randn(x_gs.size()).to(x.get_device()) * self.noise_stddev

                x    = DataAugmenter()(x)
                x_gs = DataAugmenter()(x_gs)

                x    = ClassBlender(self.blend_factor)(x)  
                x_gs = ClassBlender(self.blend_factor)(x_gs)  

            x    = x.clamp(0, 1)
            x_gs = x_gs.clamp(0, 1)

            x = self.conlayer1[k](x)
            x = self.conlayer2[k](x)
            x = self.conlayer3[k](x)
                        
            pens = []
            out  = []
            for k2 in np.arange(self.n):
                x0 = self.conlayer4[k][k2](x_gs)

                x_ = ch.cat([x0, x], dim=1)
                x_ = self.conlayer5[k][k2](x_)
                    
                x_ = x_.view(x_.shape[0], -1)
                x0 = self.dense1[k][k2](x_)

                pens += [x0]

                x1 = self.dense2[k][k2](x0)
                out += [x1]

            penultimate += [pens]
            
            if len(pens) > 1:
                penultimate2 += [ch.cat(pens, dim=-1)]
            else:
                penultimate2 += pens
            
            if len(out)>1:
                outputs += [ch.cat(out, dim=-1)]
            else:
                outputs += out

        # Permute to make sure batch-size is in first dimension
        # Ensures compatibility with DataParallel()
        return ch.stack(outputs).permute(1, 0, 2)

def predict(model, x):
    outputs = model(x)
    # Reshuffle outputs back to actual shape
    outputs = outputs.permute(1, 0, 2)
    # Run activation function on activations
    outputs = ch.tanh(outputs)
    outputs = [o for o in outputs]
    outputs = ch.cat(outputs, dim=-1)
    outputs = ch.mm(outputs, ch.t(model.module.M))
    # Log-ReLU
    logits = ch.log(relu(outputs) + 1e-6)
    return logits
