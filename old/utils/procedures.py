import os
import pickle
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch
import torch.nn.functional as F
import csv
import random
import math
import numpy as np
from sklearn import metrics
#import noise



def compute_weights(iterable):
    return [sum(iterable) / (iterable[i] * len(iterable)) if iterable[i] != 0 else float("inf") for i in range(len(iterable))]


def print_format(iterable):
    #return ["{0:.8f}".format(i) if i is not None else "{0}".format(i) for i in iterable]
    return ["{0:.8f}".format(i) if i is not float("inf") else "{0}".format(i) for i in iterable]


def probabilities(outputs):
    return F.softmax(outputs, dim=1)


def max_probabilities(outputs):
    return F.softmax(outputs, dim=1).max(dim=1)[0]


def predictions(outputs):
    #return outputs.max(dim=1)[1]
    return outputs.argmax(dim=1)


def predictions_total(outputs):
    #print(outputs.argmax(dim=1))
    return outputs.argmax(dim=1).bincount(minlength=outputs.size(1)).tolist()


def entropies(outputs):
    probabilities_log_probabilities = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
    # we could make a weighted sum to compute entropy!!! I think we should use expande_as... or boradcast??? Weighted Entropy!!!
    #probabilities_log_probabilities * weights.expand_as(probabilities_log_probabilities)
    return -1.0 * probabilities_log_probabilities.sum(dim=1)


def entropies_grads(outputs):
    entropy_grads = - (1.0 + F.log_softmax(outputs, dim=1))
    #entropy_grads = - (1.0 + F.log_softmax(outputs, dim=1)) * F.softmax(outputs, dim=1) * (1.0 - (F.softmax(outputs, dim=1)))
    return entropy_grads.sum(dim=0).tolist()


"""
def self_cross_entropies(outputs):
    # Deprecated... You should now inform the self-target... 
    return - 1.0 * F.log_softmax(outputs, dim=1)[range(outputs.size(0)), outputs.argmax(dim=1)]


def self_cross_entropies_grads(outputs):
    # Deprecated... You should now inform the self-target... 
    self_cross_entropies_grads = [0 for i in range(len(predictions_total(outputs)))]
    for i in range(len(predictions(outputs))):
        self_cross_entropies_grads[predictions(outputs)[i]] += - (1.0 / (F.softmax(outputs, dim=1)[i, predictions(outputs)[i]].item()))
        #self_cross_entropies_grads[predictions(outputs)[i]] += - (1.0 - (F.softmax(outputs, dim=1)[i, predictions(outputs)[i]].item()))
    return self_cross_entropies_grads
"""


def cross_entropies(outputs, targets):
    """ New function... """
    return - 1.0 * F.log_softmax(outputs, dim=1)[range(outputs.size(0)), targets]


def cross_entropies_grads(outputs, targets):
    """ quando tiver targets... ou self-targets... kkkkkk... """
    #cross_entropies_grads = [0 for i in range(len(predictions_total(outputs)))]
    cross_entropies_grads = [0 for i in range(outputs.size(1))]
    for i in range(len(predictions(outputs))):
        #cross_entropies_grads[predictions(outputs)[i]] += - (1.0 / (F.softmax(outputs, dim=1)[i, targets[i]].item()))
        cross_entropies_grads[predictions(outputs)[i]] += - (1.0 - (F.softmax(outputs, dim=1)[i, targets[i]].item()))
    return cross_entropies_grads


def make_equitable(outputs, criterion, weights):
    weights = torch.Tensor(weights).cuda()
    weights.requires_grad = False
    #print(weights)
    return weights[predictions(outputs)] * criterion[range(outputs.size(0))]


"""
def get_information_theory(logits):
    predictions = logits.argmax(dim=1)
    probabilities = probabilities_from_logits(logits)
    means_of_probabilities = probabilities.avg(dim=0, keepdim=True)
    entropies = entropies_from_logits(logits)
    mean_of_entropies = entropies.avg()
    entropy_of_means = entropies_from_probabilities(means_of_probabilities)
    #entropy_of_means = compute_entropies_from_logits(logits.avg(dim=0, keepdim=True))
    std_of_pred_count = predictions.bincount().float().std().item()
    return predictions, probabilities, means_of_probabilities, entropies, mean_of_entropies, entropy_of_means, std_of_pred_count
"""

"""
def entropies_from_logits(logits):
    #print(float(torch.finfo(torch.float64).eps))
    prob_log_prob = F.softmax(logits, dim=1) * torch.log(F.softmax(logits, dim=1) + float(torch.finfo(torch.float32).eps))
    #probabilities_log_probabilities = probabilities * torch.log(probabilities + 1e-32)
    return -1.0 * prob_log_prob.sum(dim=1)
"""

def entropies_from_logits(logits):
    #return -(F.softmax(logits, dim=1) * torch.log(F.softmax(logits, dim=1))).sum(dim=1)
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)

######## NEW!!!! ###########
######## NEW!!!! ###########
def entropies_from_probabilities(probabilities):
    #return -(F.softmax(logits, dim=1) * torch.log(F.softmax(logits, dim=1))).sum(dim=1)
    #return -(probabilities * torch.log(probabilities)).sum(dim=1)
    #"""
    if len(probabilities.size()) == 2:
        #print("chegou1")
        return -(probabilities * torch.log(probabilities)).sum(dim=1)
    elif len(probabilities.size()) == 3:
        print("chegou2")
        #print(probabilities.size())
        #print((probabilities * torch.log(probabilities)).sum(dim=2).size())
        #print((probabilities * torch.log(probabilities)).sum(dim=2).mean(dim=1).size())
        return -(probabilities * torch.log(probabilities)).sum(dim=2).mean(dim=1)
    #"""
######## NEW!!!! ###########
######## NEW!!!! ###########


def save_object(object, path, file):
    with open(os.path.join(path, file + '.pkl'), 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_object(path, file):
    with open(os.path.join(path, file + '.pkl'), 'rb') as f:
        return pickle.load(f)


def save_dict_list_to_csv(dict_list, path, file):
    with open(os.path.join(path, file + '.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dict_list[0].keys())
        writer.writeheader()
        for dict in dict_list:
            writer.writerow(dict)


def load_dict_list_from_csv(path, file):
    dict_list = []
    with open(os.path.join(path, file + '.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for dict in reader:
            dict_list.append(dict)
    return dict_list


class MeanMeter(object):
    """Computes and stores the current averaged current mean"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    #contingency_matrix = metrics.cluster.contingency_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def asinh(x):
    return torch.log(x+(x**2+1)**0.5)


def acosh(x):
    return torch.log(x+(x**2-1)**0.5)


def atanh(x):
    #return 0.5*torch.log((1+x)/(1-x))
    return 0.5*torch.log(((1+x)/((1-x)+0.000001))+0.000001)


def sinh(x):
    return (torch.exp(x)-torch.exp(-x))/2


def cosine_similarity(features, prototypes):
    return F.cosine_similarity(features.unsqueeze(2), prototypes.t().unsqueeze(0), dim=1, eps=1e-6)


"""
def normalized_euclidean_distances(features, prototypes, pnorm):
    #######################################################################################################################
    #######################################################################################################################
    #diff = features.unsqueeze(2) - prototypes.t().unsqueeze(0)
    #diff2 = features.t().unsqueeze(0) - prototypes.unsqueeze(2)
    #print("F.PAIRWISE:\n", F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=2.0))
    #print("DIFF*DIFF2:\n", torch.sqrt(torch.diagonal(torch.matmul(diff2.permute(2, 0, 1), diff), offset=0, dim1=1, dim2=2)))
    #print(math.sqrt(features.size(1)))
    #######################################################################################################################
    #######################################################################################################################
    return F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=pnorm)
    # THE CORRECTION BELLOW MAY MAKE THE SOLUTION WORKS BETTER FOR ALL MODELS WITH DIFERENT SIZES OF FEATURE VECTORS!!!
    # ACTUALLY, THE CORRECTION BELLOW IS POISON!!! IT BREAK THE TRAINING IN SOME CASES!!! PLEASE, NEVER USE THIS!!!!!!!
    #return 20*F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=pnorm) / math.sqrt(features.size(1))
"""


def euclidean_distances(features, prototypes, pnorm):
    #return F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=pnorm, eps=1e-3)
    """
    print(features.size())
    print(features.unsqueeze(2).size())
    print(features.unsqueeze(2).unsqueeze(3).size())
    print("###########")
    print(prototypes.size())
    print(prototypes.permute(0, 2, 1).unsqueeze(0).size())
    print(prototypes.unsqueeze(0).permute(0, 3, 1, 2).size())
    print("###########")
    """
    #############################################################################################
    if len(prototypes.size()) == 2:
        return F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=pnorm)
    #############################################################################################
    #print(F.pairwise_distance(features.unsqueeze(2).unsqueeze(3), prototypes.unsqueeze(0).permute(0, 3, 1, 2), p=pnorm).size())
    #print(features.unsqueeze(2).unsqueeze(3).size())
    #print(prototypes.unsqueeze(0).permute(0, 3, 1, 2).size())
    #print(prototypes.unsqueeze(0).permute(0, 2, 3, 1).size())
    #############################################################################################
    else:
        #print("multiprototypes!!!")
        #return F.pairwise_distance(features.unsqueeze(2).unsqueeze(3), prototypes.unsqueeze(0).permute(0, 3, 1, 2), p=pnorm).mean(dim=1)
        #return F.pairwise_distance(features.unsqueeze(2).unsqueeze(3), prototypes.unsqueeze(0).permute(0, 3, 1, 2), p=pnorm).sum(dim=1)
        return F.pairwise_distance(features.unsqueeze(2).unsqueeze(3), prototypes.unsqueeze(0).permute(0, 3, 1, 2), p=pnorm)


"""
def logits_from_distances(distances, guage):
    #print("very ok!!!")
    #print(distances)
    if guage.startswith("ga"):
        print("Entrou no GA...")
        beta = float(guage.strip("ga")) 
        #### 20 DOES NOT WORK!!!! 10 LOW ODD PERFORMANCE!!! TRY 1 OR 5???
        print("Beta:", beta)
        #### ====>>>> THE BELLOW INDEED AFFECTS BOTH PROBABILITIES AND GRADIENTES!!!!!!!!!!!
        #### ====>>>> THE BELLOW INDEED AFFECTS BOTH PROBABILITIES AND GRADIENTES!!!!!!!!!!!
        #### ====>>>> BUT YOU CAN STILL PLAY WITH THE BETA PARAMETER!!!!!!!!
        #### ====>>>> BUT YOU CAN STILL PLAY WITH THE BETA PARAMETER!!!!!!!!
        logits = beta * (distances/distances.mean(dim=1).unsqueeze(1).detach()) #### option 1
    elif guage.startswith("gb"):
        print("Entrou no GB...")
        beta = float(guage.strip("gb")) 
        ### APPEARS TO WORK WITH 10, 20 AND 30!!! TRY 0 OR 5???
        print("Beta:", beta)
        #### ====>>>> THE BELLOW DOES NOT AFFECT PROBABILITIES NEITHER GRADIENTES!!!!!!!!!!!
        #### ====>>>> THE BELLOW DOES NOT AFFECT PROBABILITIES NEITHER GRADIENTES!!!!!!!!!!!
        #### ====>>>> AND YOU CAN STILL PLAY WITH THE BETA PARAMETER!!!!!!!!
        #### ====>>>> AND YOU CAN STILL PLAY WITH THE BETA PARAMETER!!!!!!!!
        logits = (distances - distances.mean(dim=1).unsqueeze(1).detach()) + beta #### option 2
    ####logits = beta * (distances/distances.mean(dim=0).detach())
    ####logits = (distances - distances.mean(dim=0).detach()) + beta
    #print(logits)
    return logits
"""

def mahalanobis_distances(features, prototypes, precisions):
    diff = features.unsqueeze(2) - prototypes.t().unsqueeze(0)
    diff2 = features.t().unsqueeze(0) - prototypes.unsqueeze(2)
    precision_diff = torch.matmul(precisions.unsqueeze(0), diff)
    extended_product = torch.matmul(diff2.permute(2, 0, 1), precision_diff)
    mahalanobis_square = torch.diagonal(extended_product, offset=0, dim1=1, dim2=2)
    mahalanobis = torch.sqrt(mahalanobis_square)
    #mahalanobis = mahalanobis.div(math.sqrt(prototypes.detach().size(1)))
    #print(F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=2.0))
    return mahalanobis


def multiprecisions_mahalanobis_distances(features, prototypes, multiprecisions):
    mahalanobis_square = torch.Tensor(features.size(0), prototypes.size(0)).cuda()
    for prototype in range(prototypes.size(0)):
        diff = features - prototypes[prototype]
        multiprecisions.unsqueeze(0)
        diff.unsqueeze(2)
        precision_diff = torch.matmul(multiprecisions.unsqueeze(0), diff.unsqueeze(2))
        product = torch.matmul(diff.unsqueeze(1), precision_diff).squeeze()
        mahalanobis_square[:, prototype] = product
    mahalanobis = torch.sqrt(mahalanobis_square)
    #mahalanobis = mahalanobis.div(math.sqrt(prototypes.detach().size(1)))
    #print(F.pairwise_distance(features.unsqueeze(2), prototypes.t().unsqueeze(0), p=2.0))
    return mahalanobis

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    #"""
    print("calling randbox")
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    #"""

    """
    r = 0.5 + np.random.rand(1)/2
    s = 0.5/r
    if np.random.rand(1) < 0.5:
        r, s = s, r
    #print(r)
    #print(s)
    #print(r * s)
    cut_w = np.int(W * r)
    cut_h = np.int(H * s)
    """

    ####cx = np.random.randint(W)
    ####cy = np.random.randint(H)
    cx = np.random.randint(cut_w // 2, high=W - cut_w // 2)
    cy = np.random.randint(cut_h // 2, high=H - cut_h // 2)
    #print(cx)
    #print(cy)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    #print(bbx1)
    #print(bbx2)
    #print(bby1)
    #print(bby2)

    return bbx1, bby1, bbx2, bby2


def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    #print("-" * 36)
    print("Total number of parameters: {:.2e}".format(total_num_params))


"""
import matplotlib.pylab as plt
%matplotlib inline

d = torch.linspace(-10.0, 10.0)
s = Swish()
res = s(d)
res2 = torch.relu(d)

plt.title("Swish transformation")
plt.plot(d.numpy(), res.numpy(), label='Swish')
plt.plot(d.numpy(), res2.numpy(), label='ReLU')
plt.legend()
"""

############################################
def perlin(x,y,seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u) # FIX1: I was using n10 instead of n01
    return lerp(x1,x2,v) # FIX2: I also had to reverse x1 and x2 here

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

def perlin_noise(image_size=32,slice_size=64):
    #seed = random.randint(0, 1000000)
    #lin = np.linspace(0,5,image_size,endpoint=False)
    #x,y = np.meshgrid(lin,lin) # FIX3: I thought I had to invert x and y here but it was a mistake
    noise_array = np.ndarray(shape=(slice_size,3,image_size,image_size))
    for k in range(slice_size):
        ##noise_array[k][0] = ( np.random.rand(1) * perlin(x,y,seed=seed) + 1 ) / 2.0
        ##noise_array[k][1] = ( np.random.rand(1) * perlin(x,y,seed=seed) + 1 ) / 2.0
        ##noise_array[k][2] = ( np.random.rand(1) * perlin(x,y,seed=seed) + 1 ) / 2.0
        #noise_array[k][0] = ( perlin(x,y,seed=seed) + 1 ) / 2.0
        #noise_array[k][1] = ( perlin(x,y,seed=seed) + 1 ) / 2.0
        #noise_array[k][2] = ( perlin(x,y,seed=seed) + 1 ) / 2.0
        shape = (image_size,image_size)
        scale = 8
        octs = 4
        pers = 0.5
        lac = 2
        seed = np.random.randint(0,10000)
        for l in range(3):
            world = np.zeros(shape)
            for i in range(image_size):
                for j in range(image_size):
                    world[i][j] = ((np.random.rand(1) * noise.pnoise2(i/scale,j/scale, octaves=octs, persistence=pers, lacunarity=lac,
                    repeatx=image_size, repeaty=image_size, base=seed))+1)/2.0
            noise_array[k][l] = world
        #noise_array[k][0] = ( perlin(x,y,seed=seed) + 1 ) / 2.0
        #noise_array[k][1] = ( perlin(x,y,seed=seed) + 1 ) / 2.0
        #noise_array[k][2] = ( perlin(x,y,seed=seed) + 1 ) / 2.0

    return torch.from_numpy(noise_array).float()


#plt.imshow(perlin(x,y,seed=1),origin='upper')
############################################

"""
blue = [65,105,225]
green = [34,139,34]
beach = [238, 214, 175]
snow = [255, 250, 250]
mountain = [139, 137, 137]

def add_color(world):
    color_world = np.zeros(world.shape+(3,))
    for i in range(shape[0]):
        for j in range(shape[1]):
            if world[i][j] < -0.05:
                color_world[i][j] = blue
            elif world[i][j] < 0:
                color_world[i][j] = beach
            elif world[i][j] < .20:
                color_world[i][j] = green
            elif world[i][j] < 0.35:
                color_world[i][j] = mountain
            elif world[i][j] < 1.0:
                color_world[i][j] = snow

    return color_world

color_world = add_color(world)
toimage(color_world).show()
"""
