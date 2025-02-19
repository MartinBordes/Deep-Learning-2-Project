import numpy as np
import matplotlib.pyplot as plt
from principal_RBM_alpha import RBM
import copy

class DBN:
    def __init__(self, p):
        L = len(p) - 1
        self.dbn = [RBM(p[l], p[l+1]) for l in range(L)]
        self.L = L
        # self.p = p
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x)) 
    
    def train(self, X, learning_rate, len_batch, n_epochs, verbose=False):
        H = copy.deepcopy(X)
        for l in range(self.L):
            rbm = self.dbn[l]
            rbm.train(H, learning_rate, len_batch, n_epochs, verbose=False)
            # H = (np.random.rand(len(X), self.p[l+1]) < self.sigmoid(rbm.entree_sortie(H)))*1
            H = self.sigmoid(rbm.entree_sortie(H))
    
    def generer_image(self, nb_images, nb_iter, size_img, display=True):
        #images = self.dbn[-1].generer_image(nb_images, nb_iter, size_img, display=False)
        p,q=self.dbn[-1].p, self.dbn[-1].q
        images = []

        for _ in range(nb_images):
            v=(np.random.rand(p)<0.5)*1

            for _ in range(nb_iter):
                h = (np.random.rand(q)<self.sigmoid(self.dbn[-1].entree_sortie(v)))*1
                v = (np.random.rand(p)<self.sigmoid(self.dbn[-1].sortie_entree(h)))*1

            images.append(v)

        for l in reversed(range(self.L-1)):
            for k in range(nb_images):
                images[k] = (np.random.rand(self.dbn[l].p) < self.sigmoid(self.dbn[l].sortie_entree(images[k])))*1

        if display:
            plt.figure(figsize=(25,10))
            cols = 8
            rows = int(nb_images/cols) + 1
            for k in range(nb_images):
                image = images[k]
                image = image.reshape(size_img)
                plt.subplot(rows, cols, k+1)      
                plt.imshow(image, cmap='gray')
                plt.axis('off')
        
        return images
