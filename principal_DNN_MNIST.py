import numpy as np
from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM

class DNN:
    def __init__(self, p):
        # self.dbn = DBN(p)
        self.pre_dbn = DBN(p[:-1])
        self.output_layer = RBM(p[-2], p[-1])
        self.num_classes = p[-1]
        self.L = len(p) - 1
        
    def pretrain(self, X, learning_rate, len_batch, n_epochs, verbose=False):
        self.pre_dbn.train(X, learning_rate, len_batch, n_epochs, verbose)
        return self
    
    def from_pre_trained_DBN(self, dbn):
        self.pre_dbn = dbn
    
    def calcul_softmax(self, activations):
        exp_activations = np.exp(activations*np.log(1.01))
        softmax_probs = exp_activations / np.sum(exp_activations, axis=1, keepdims=True)
    
        return softmax_probs
    
    def entree_sortie_reseau(self, X):
        all_z = []
        all_h = []
    
        sortie_couche = X
        all_h.append(sortie_couche)
        
        for i in range(self.L - 1):
            rbm = self.pre_dbn.dbn[i]
            sortie_couche = rbm.entree_sortie(sortie_couche)
            all_z.append(sortie_couche)
            sortie_couche = self.relu(sortie_couche)
            all_h.append(sortie_couche)

        rbm = self.output_layer
        sortie_couche = rbm.entree_sortie(sortie_couche)
        all_z.append(sortie_couche)
        sortie_couche = self.calcul_softmax(sortie_couche)
        all_h.append(sortie_couche)
        
        return all_z, all_h
    
    def relu(self, x):
        """Compute the value of the Rectified Linear Unit activation function"""
        return x * (x > 0)
    
    def d_relu(self, x):
        """Compute the derivative of the Rectified Linear Unit activation function"""
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def retropropagation(self, X, y, learning_rate, len_batch, n_epochs):
        losses_all = []

        for epoch in range(n_epochs):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            losses = []

            for i_batch in range(0, len(X), len_batch):
                # Sélection du mini-batch
                X_batch = X_shuffled[i_batch:i_batch+len_batch]
                y_batch = y_shuffled[i_batch:i_batch+len_batch]

                # Passe avant (forward pass)
                all_z, all_h = self.entree_sortie_reseau(X_batch)

                # Calcul de la perte (loss) - entropie croisée
                # loss = -np.mean(np.sum(y_batch * np.log(output_probs), axis=1))
                loss = -np.mean(np.log(all_h[-1][np.arange(len(y_batch)), y_batch]))
                losses.append(loss)

                # Rétropropagation (backpropagation)
                num_samples = len(y_batch)
                one_hot_y_batch = np.zeros((num_samples, self.num_classes))
                one_hot_y_batch[np.arange(num_samples), y_batch] = 1

                grad_z = all_h[-1] - one_hot_y_batch

                list_rbm = self.pre_dbn.dbn + [self.output_layer]
                    
                # Mise à jour des poids et des biais pour chaque RBM
                for i in range(self.L-1, 0, -1):
                    rbm = list_rbm[i]  # Get the current RBM
                    grad_W = (1 / num_samples) * all_h[i].T @ grad_z 
                    grad_b = (1 / num_samples) * np.sum(grad_z)
                    grad_h = grad_z @ rbm.W.T
                    grad_z = grad_h * self.d_relu(all_z[i-1])

                    rbm.b -= learning_rate * grad_b
                    rbm.W -= learning_rate * grad_W

                rbm = list_rbm[0]
                grad_W = (1 / num_samples) * all_h[0].T @ grad_z 
                grad_b = (1 / num_samples) * np.sum(grad_z)

                rbm.b -= learning_rate * grad_b
                rbm.W -= learning_rate * grad_W
            
            erreur, _ = self.test(X, y)
            mean_loss = np.mean(losses)
            losses_all.append(mean_loss)

            # Affichage de la perte à la fin de chaque epoch
            print(f"Epoch {epoch+1}/{n_epochs}, Erreur : {round(erreur*100,2)} %, Loss: {round(mean_loss,3)}")

        return losses
    
    def test(self, X, y):
        _, all_h = self.entree_sortie_reseau(X)
        y_hat = all_h[-1]
        y_pred = np.argmax(y_hat, axis=1)

        erreurs = (y_pred != y).sum()
        taux_erreur = erreurs / len(y)

        return taux_erreur, y_hat
    
