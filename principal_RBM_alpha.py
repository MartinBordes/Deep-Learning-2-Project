import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


class RBM:
    def __init__(self, p,q):
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(size=(p,q))*np.sqrt(10**(-2))
        self.q = q
        self.p = p
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x)) 
        
    def entree_sortie(self,V): 
        return V@self.W+self.b
        
    def sortie_entree(self,H):
        return H@self.W.T+self.a
    
    def train(self, X, learning_rate, len_batch, n_epochs, verbose=False, display = False):
        p,q=self.W.shape
        losses = []

        for i in tqdm(range(n_epochs)):
            np.random.shuffle(X)
            n=X.shape[0]

            for i_batch in range(0, n ,len_batch):
                X_batch=X[i_batch:min(i_batch+len_batch,n),:]
                t_batch_i = X_batch.shape[0] 

                pH_V0 = self.sigmoid(self.entree_sortie(X_batch))
                H0 = (np.random.rand(t_batch_i,q) < pH_V0)*1 
                pV_H0 = self.sigmoid(self.sortie_entree(H0))
                V1 = (np.random.rand(t_batch_i,p) < pV_H0)*1 
                pH_V1 = self.sigmoid(self.entree_sortie(V1))

                da = np.sum(X_batch-V1,axis=0)
                db = np.sum(pH_V0-pH_V1,axis=0)
                dW = X_batch.T@pH_V0 - V1.T@pH_V1

                self.a += learning_rate * da
                self.b += learning_rate * db
                self.W += learning_rate * dW
            
            H = self.sigmoid(self.entree_sortie(X))
            X_rec = self.sigmoid(self.sortie_entree(H))
            loss = np.mean((X-X_rec)**2)
            losses.append(loss)
            if i%10 == 0 and verbose:
              print(f"Epoch  {i} / {n_epochs} - loss : {loss}")
     
        if display :
            plt.plot(losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Évolution de la loss')
            plt.show()
            print("Loss finale :", losses[-1])

    def generer_image(self, nb_images, nb_iter, size_img, display = True):
        p,q=self.W.shape
        images = []

        for _ in range(nb_images):
            v=(np.random.rand(p)<0.5)*1

            for _ in range(nb_iter):
                h = (np.random.rand(q)<self.entree_sortie(v))*1
                v = (np.random.rand(p)<self.sortie_entree(h))*1

            v=v.reshape(size_img)
            images.append(v)

        if display:
            plt.figure(figsize=(25,10))
            cols = 8
            rows = int(nb_images/cols) + 1
            for k in range(nb_images):
                image = images[k]
                plt.subplot(rows, cols, k+1)      
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            
        return images