import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import random as r


class SVM_classifier:
    def __init__(self, learning_rate, no_of_iterations, lambda_parameters, gamma,d, kernel='rbf'):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameters = lambda_parameters
        self.gamma = gamma
        self.kernel = kernel
        self.iteration = 0  
        self.d = d
        self.kernel_functions = {
            'linear': self.linear_kernel,
            'poly': self.polynomial_kernel,
            'rbf': self.rbf_kernel,
            'laplace': self.laplace_kernel,
            'matern': self.matern_kernel,
            'distance_based': self.distance_based_kernel,
            'cosine': self.cosine_kernel,
            'sobolev': self.sobolev_kernel,
            'product' : self.product_operator_kernel
        }
        self.kernel_matrix = 0

    def fit(self, X, Y,X_test,Y_test, resume=False):
        self.m, self.n = X.shape
        self.alpha = np.zeros(self.m) if not resume else self.alpha
        self.b = 0 if not resume else self.b
        self.X = X
        self.Y = np.where(Y <= 0, -1, 1)
        self.X_test = X_test
        self.Y_test = np.where(Y_test <= 0, -1, 1)

        self.losses = [] if not resume else self.losses
        #if not self.no_of_iterations>=self.iteration-1:
        self.kernel_matrix = np.array([[self.kernel_functions[self.kernel](self.X[i], self.X[j]) for j in range(self.m)] for i in range(self.m)])

        for i in range(self.iteration, self.no_of_iterations):
            self.update_weights()
            loss = 1#self.compute_loss()
            self.losses.append(loss)
            print(f"Iteration {i}/{self.no_of_iterations}, Loss: {loss}")
            self.learning_rate *= 0.9
            self.lambda_parameters *= 0.9
            self.iteration += 1
            
            #for _ in range(20):
                #dropout
                #self.alpha[r.randint(0,len(self.alpha)-1)]

            
        self.save_state()
        
        

    def save_state(self):
        state = {
            'X': self.X,
            'Y': self.Y,
            'alpha': self.alpha,
            'b': self.b,
            'losses': self.losses,
            'iteration': self.iteration,
            'learning_rate': self.learning_rate,
            'lambda_parameters': self.lambda_parameters
        }
        joblib.dump(state, 'SVM/model/svm_'+self.kernel+'_linear_model_state.joblib')
        print("État du modèle sauvegardé.")

    def load_state(self, X, Y):
        state = joblib.load('SVM/model/svm_'+self.kernel+'_linear_model_state.joblib')
        self.alpha = state['alpha']
        self.b = state['b']
        self.losses = state['losses']
        self.iteration = state['iteration']
        self.learning_rate = state['learning_rate']
        self.lambda_parameters = state['lambda_parameters']
        self.m, self.n = X.shape
        self.X = X
        self.Y = np.where(Y <= 0, -1, 1)
        print("État du modèle chargé.")

    def load_params(self):
        state = joblib.load('SVM/model/svm_'+self.kernel+'_linear_model_state.joblib')
        self.alpha = state['alpha']
        self.b = state['b']
        self.losses = state['losses']
        self.iteration = state['iteration']
        self.learning_rate = state['learning_rate']
        self.lambda_parameters = state['lambda_parameters']
        self.X = state["X"]
        self.Y = state["Y"]
        self.m, self.n = self.X.shape


    def linear_kernel(self, x1, x2):
        return self.d * np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=2, c=1):
        return (np.dot(x1, x2) + c) ** degree

    def rbf_kernel(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

    def laplace_kernel(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) / sigma)

    def matern_kernel(self, x1, x2, theta=1):
        distance = np.linalg.norm(x1 - x2)
        return (1 + (np.sqrt(5) * distance) / theta + (5 * distance ** 2) / (3 * theta ** 2)) * np.exp(-np.sqrt(5) * distance / theta)

    def distance_based_kernel(self, x1, x2):
        return self.d * (np.linalg.norm(x1) + np.linalg.norm(x2) - np.linalg.norm(x1 - x2))

    def cosine_kernel(self, x1, x2):
        return np.cos(np.dot(x1, x2))

    def sobolev_kernel(self, x1, x2):
        def k1(x):
            return x - 0.5
        def k2(x):
            return (k1(x) - 0.5) ** 2
        def k4(x):
            return (k1(x) ** 4 - k1(x) ** 2 / 2 + 7 / 240) / 24
        return k1(x1) * k1(x2) + k2(x1) * k2(x2) - k4(np.abs(x1 - x2))


    # Kernel using product operator (Tensorized Kernel)
    def product_operator_kernel(self,x1, x2):
        result = 1
        for xi, xj in zip(x1, x2):
            result *= self.rbf_kernel(xi, xj)
        return result

    # Kernel using sum operator
    def sum_operator_kernel(x1, x2, one_dimensional_kernel):
        result = 0
        for xi, xj in zip(x1, x2):
            result += one_dimensional_kernel(xi, xj)
        return result


    def update_weights(self):
        
        for i in range(self.m):
            sum_term = np.sum(self.alpha * self.Y * self.kernel_matrix[i, :])
            condition = self.Y[i] * (sum_term + self.b) >= 0.3

            if condition:
                
                self.alpha[i] -= self.learning_rate * (2 * self.lambda_parameters * self.alpha[i])
            else:
                penalty = 10 if self.Y[i] == 1 else 1
                self.alpha[i] += self.learning_rate * penalty * (1 - self.Y[i] * (sum_term + self.b))
                self.b += self.learning_rate * self.Y[i] * penalty
    
    def temp_decision_function(self,x,alpha, bias):
        
        result = 0
        for i in range(self.m):
            result += alpha[i] * self.Y[i] * self.kernel_functions[self.kernel](self.X[i], x)
        return result + bias
    def update_weights_dynamique(self):


        def temp_predict(X, alpha, bias, n_jobs=None):
            with Pool(processes=n_jobs) as pool:
                params = [(x, alpha, bias) for x in X]
                output = pool.starmap(self.temp_decision_function, params)
            
            output = np.array(output)  # Conversion de la liste en tableau NumPy
            predicted_labels = np.sign(output)
            y_hat = predicted_labels#np.where(predicted_labels == -1, 0, 1)
            return y_hat

        best_condition  = 0.1
        best_precision = 0
        for condition_test in range(1,10,2):
            condition_test/=10
            temp_alpha = self.alpha
            temp_bias = self.b
            for i in range(self.m):
                sum_term = np.sum(temp_alpha * self.Y * self.kernel_matrix[i, :])
                condition = self.Y[i] * (sum_term + temp_bias) >= condition_test

                if condition:
                    temp_alpha[i] -= self.learning_rate * (2 * self.lambda_parameters * temp_alpha[i])
                else:
                    penalty = 10 if self.Y[i] == 1 else 1
                    temp_alpha[i] += self.learning_rate * penalty * (1 - self.Y[i] * (sum_term + temp_bias))
                    temp_bias += self.learning_rate * self.Y[i] * penalty
            precision = accuracy_score(self.Y_test, temp_predict(self.X_test,temp_alpha,temp_bias))
            print("Condition : "+str(condition_test)+" , precision : "+str(precision))
            if precision>best_precision:
                best_precision = precision
                best_condition = condition_test
        print("Using condition : " + str(best_condition))
        for i in range(self.m):
            sum_term = np.sum(self.alpha * self.Y * self.kernel_matrix[i, :])
            condition = self.Y[i] * (sum_term + self.b) >= best_condition
            if condition:
                self.alpha[i] -= self.learning_rate * (2 * self.lambda_parameters * self.alpha[i])
            else:
                penalty = 10 if self.Y[i] == 1 else 1
                self.alpha[i] += self.learning_rate * penalty * (1 - self.Y[i] * (sum_term + self.b))
                self.b += self.learning_rate * self.Y[i] * penalty 

    def decision_function(self, x):

       # if x.ndim == 1:
       #     x = x.reshape(1, -1)  # Convertir en tableau 2D avec une seule ligne
        result = 0
        for i in range(self.m):
            result += self.alpha[i] * self.Y[i] * self.kernel_functions[self.kernel](self.X[i], x)
        return result + self.b

    def predict(self, X, threshold=0, n_jobs=None):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        with Pool(processes=n_jobs) as pool:
            output = pool.map(self.decision_function, X)
        
        output = np.array(output)  # Conversion de la liste en tableau NumPy
        predicted_labels = np.sign(output - threshold)
        y_hat = np.where(predicted_labels == -1, 0, 1)
        return y_hat
    
    def predire_instance_unique(classif_svm, x):
        """
        Fonction pour faire une prédiction sur une seule instance (tableau unidimensionnel).
        
        :param classif_svm: Modèle SVM déjà entraîné
        :param x: Tableau unidimensionnel représentant une instance de données.
        :return: Prédiction pour l'instance fournie.
        """
        # Vérifier si le tableau est unidimensionnel
        if x.ndim == 1:
            # Redimensionner le tableau pour qu'il soit compatible avec la prédiction (comme une matrice d'une ligne)
            x = x.reshape(1, -1)
        
        # Faire la prédiction avec le modèle SVM
        prediction = classif_svm.predict(x)
        
        return prediction



    def compute_loss(self):
        loss = 0
        for i in range(self.m):
            sum_term = 0
            for j in range(self.m):
                sum_term += self.alpha[j] * self.Y[j] * self.kernel_functions[self.kernel](self.X[j], self.X[i])
            loss += max(0, 1 - self.Y[i] * (sum_term + self.b))
        loss += self.lambda_parameters * np.sum(self.alpha ** 2)
        return loss

    def plot_loss(self):
        plt.plot(self.losses)
        plt.title('Loss over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

    def find_best_threshold(self, X_test,Y_test ,thresholds):

        Y_test = np.where(Y_test <= 0, -1, 1)
        output = np.array([self.decision_function(x) for x in X_test])
        # Plotting the outputs with colors based on Y_test
        colors = ['red' if y == -1 else 'green' for y in Y_test]

        plt.scatter(range(len(output)), output, c=colors)
        plt.xlabel('Index')
        plt.ylabel('Output')
        plt.title('Outputs colored by Y_test values')
        plt.show()
        best_threshold = thresholds[0]
        best_accuracy = 0
        
        for threshold in thresholds:
            predicted_labels = np.sign(output - threshold)
            predictions = np.where(predicted_labels >= threshold, 1, -1)
            accuracy = accuracy_score(Y_test, predictions)
            print(f"Treshold: {threshold}, accuracy: {accuracy}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                
        return best_threshold, best_accuracy

class SVM_classifier_tool:
    def __init__(self, learning_rate, no_of_iterations, lambda_parameters, gamma,d, kernel='rbf'):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameters = lambda_parameters
        self.gamma = gamma
        self.kernel = kernel
        self.iteration = 0  
        self.d = d
        self.kernel_functions = {
            'linear': self.linear_kernel,
            'poly': self.polynomial_kernel,
            'rbf': self.rbf_kernel,
            'laplace': self.laplace_kernel,
            'matern': self.matern_kernel,
            'distance_based': self.distance_based_kernel,
            'cosine': self.cosine_kernel,
            'sobolev': self.sobolev_kernel,
            'product' : self.product_operator_kernel
        }
        self.kernel_matrix = 0

    def fit(self, X, Y,X_test,Y_test, resume=False):
        self.m, self.n = X.shape
        self.alpha = np.zeros(self.m) if not resume else self.alpha
        self.b = 0 if not resume else self.b
        self.X = X
        self.Y = np.where(Y <= 0, -1, 1)
        self.X_test = X_test
        self.Y_test = np.where(Y_test <= 0, -1, 1)

        self.losses = [] if not resume else self.losses
        #if not self.no_of_iterations>=self.iteration-1:
        self.kernel_matrix = np.array([[self.kernel_functions[self.kernel](self.X[i], self.X[j]) for j in range(self.m)] for i in range(self.m)])

        for i in range(self.iteration, self.no_of_iterations):
            self.update_weights()
            loss = 1#self.compute_loss()
            self.losses.append(loss)
            print(f"Iteration {i}/{self.no_of_iterations}, Loss: {loss}")
            self.learning_rate *= 0.9
            self.lambda_parameters *= 0.9
            self.iteration += 1
            
            #for _ in range(20):
                #dropout
                #self.alpha[r.randint(0,len(self.alpha)-1)]

            
        
        
    
    def linear_kernel(self, x1, x2):
        return self.d * np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=2, c=1):
        return (np.dot(x1, x2) + c) ** degree

    def rbf_kernel(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

    def laplace_kernel(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) / sigma)

    def matern_kernel(self, x1, x2, theta=1):
        distance = np.linalg.norm(x1 - x2)
        return (1 + (np.sqrt(5) * distance) / theta + (5 * distance ** 2) / (3 * theta ** 2)) * np.exp(-np.sqrt(5) * distance / theta)

    def distance_based_kernel(self, x1, x2):
        return self.d * (np.linalg.norm(x1) + np.linalg.norm(x2) - np.linalg.norm(x1 - x2))

    def cosine_kernel(self, x1, x2):
        return np.cos(np.dot(x1, x2))

    def sobolev_kernel(self, x1, x2):
        def k1(x):
            return x - 0.5
        def k2(x):
            return (k1(x) - 0.5) ** 2
        def k4(x):
            return (k1(x) ** 4 - k1(x) ** 2 / 2 + 7 / 240) / 24
        return k1(x1) * k1(x2) + k2(x1) * k2(x2) - k4(np.abs(x1 - x2))


    # Kernel using product operator (Tensorized Kernel)
    def product_operator_kernel(self,x1, x2):
        result = 1
        for xi, xj in zip(x1, x2):
            result *= self.rbf_kernel(xi, xj)
        return result

    # Kernel using sum operator
    def sum_operator_kernel(x1, x2, one_dimensional_kernel):
        result = 0
        for xi, xj in zip(x1, x2):
            result += one_dimensional_kernel(xi, xj)
        return result


    def update_weights(self):
        
        for i in range(self.m):
            sum_term = np.sum(self.alpha * self.Y * self.kernel_matrix[i, :])
            condition = self.Y[i] * (sum_term + self.b) >= 0.3

            if condition:
                
                self.alpha[i] -= self.learning_rate * (2 * self.lambda_parameters * self.alpha[i])
            else:
                penalty = 1 if self.Y[i] == 1 else 1
                self.alpha[i] += self.learning_rate * penalty * (1 - self.Y[i] * (sum_term + self.b))
                self.b += self.learning_rate * self.Y[i] * penalty
    
    def temp_decision_function(self,x,alpha, bias):
        
        result = 0
        for i in range(self.m):
            result += alpha[i] * self.Y[i] * self.kernel_functions[self.kernel](self.X[i], x)
        return result + bias
    def update_weights_dynamique(self):


        def temp_predict(X, alpha, bias, n_jobs=None):
            with Pool(processes=n_jobs) as pool:
                params = [(x, alpha, bias) for x in X]
                output = pool.starmap(self.temp_decision_function, params)
            
            output = np.array(output)  # Conversion de la liste en tableau NumPy
            predicted_labels = np.sign(output)
            y_hat = predicted_labels#np.where(predicted_labels == -1, 0, 1)
            return y_hat

        best_condition  = 0.1
        best_precision = 0
        for condition_test in range(1,10,2):
            condition_test/=10
            temp_alpha = self.alpha
            temp_bias = self.b
            for i in range(self.m):
                sum_term = np.sum(temp_alpha * self.Y * self.kernel_matrix[i, :])
                condition = self.Y[i] * (sum_term + temp_bias) >= condition_test

                if condition:
                    temp_alpha[i] -= self.learning_rate * (2 * self.lambda_parameters * temp_alpha[i])
                else:
                    penalty = 10 if self.Y[i] == 1 else 1
                    temp_alpha[i] += self.learning_rate * penalty * (1 - self.Y[i] * (sum_term + temp_bias))
                    temp_bias += self.learning_rate * self.Y[i] * penalty
            precision = accuracy_score(self.Y_test, temp_predict(self.X_test,temp_alpha,temp_bias))
            print("Condition : "+str(condition_test)+" , precision : "+str(precision))
            if precision>best_precision:
                best_precision = precision
                best_condition = condition_test
        print("Using condition : " + str(best_condition))
        for i in range(self.m):
            sum_term = np.sum(self.alpha * self.Y * self.kernel_matrix[i, :])
            condition = self.Y[i] * (sum_term + self.b) >= best_condition
            if condition:
                self.alpha[i] -= self.learning_rate * (2 * self.lambda_parameters * self.alpha[i])
            else:
                penalty = 10 if self.Y[i] == 1 else 1
                self.alpha[i] += self.learning_rate * penalty * (1 - self.Y[i] * (sum_term + self.b))
                self.b += self.learning_rate * self.Y[i] * penalty 

    def decision_function(self, x):
        result = 0
        for i in range(self.m):
            result += self.alpha[i] * self.Y[i] * self.kernel_functions[self.kernel](self.X[i], x)
        return result + self.b

    def predict(self, X, threshold=0, n_jobs=None):
        with Pool(processes=n_jobs) as pool:
            output = pool.map(self.decision_function, X)
        
        output = np.array(output)  # Conversion de la liste en tableau NumPy
        predicted_labels = np.sign(output - threshold)
        y_hat = np.where(predicted_labels == -1, 0, 1)
        return y_hat
    
  


class SVM_multiclassifier:
    def __init__(self, learning_rate, no_of_iterations, lambda_parameters, gamma, d, kernel, multiclass_strategy):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameters = lambda_parameters
        self.gamma = gamma
        self.kernel = kernel
        self.d = d
        self.kernel_functions = {
            'linear': self.linear_kernel,
            'poly': self.polynomial_kernel,
            'rbf': self.rbf_kernel,
            'laplace': self.laplace_kernel,
            'matern': self.matern_kernel,
            'distance_based': self.distance_based_kernel,
            'cosine': self.cosine_kernel,
            'sobolev': self.sobolev_kernel,
            'product': self.product_operator_kernel
        }
        self.kernel_matrix = 0
        self.multiclass_strategy = multiclass_strategy  # Either 'ovr' (One-vs-Rest) or 'ovo' (One-vs-One)
        self.models = []  # Will store binary classifiers for multiclass scenarios

    def fit(self, X, Y, X_test, Y_test):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test

        unique_classes = np.unique(Y)
        self.models = []  # To store individual SVMs for each class (for OvR) or class pair (for OvO)

        if self.multiclass_strategy == 'ovr':
            # Train one SVM per class
            for class_label in unique_classes:
                print(f"Training SVM for class {class_label} vs Rest")
                binary_Y = np.where(Y == class_label, 1, -1)  # 1 for the current class, -1 for the rest
                svm_model = SVM_classifier(
                    self.learning_rate, self.no_of_iterations, self.lambda_parameters,
                    self.gamma, self.d, self.kernel
                )
                svm_model.fit(X, binary_Y, X_test, np.where(Y_test == class_label, 1, -1), resume=False)
                self.models.append((svm_model, class_label))

        elif self.multiclass_strategy == 'ovo':
            
            # Train one SVM for each pair of classes
            for i, class_1 in enumerate(unique_classes):
                for class_2 in unique_classes[i+1:]:
                    print(f"Training SVM for class {class_1} vs class {class_2}")
                    binary_indices = np.where((Y == class_1) | (Y == class_2))[0]  
                    binary_X = X[binary_indices] 
                    binary_Y = Y.iloc[binary_indices]  
                    binary_Y = np.where(binary_Y == class_1, 1, -1)  # 1 pour class_1, -1 pour class_2

                    svm_model = SVM_classifier_tool(
                        self.learning_rate, self.no_of_iterations, self.lambda_parameters,
                        self.gamma, self.d, self.kernel
                    )
                    svm_model.fit(binary_X, binary_Y, X_test, binary_Y, resume=False)
                    self.models.append((svm_model, class_1, class_2))

        else:
            raise ValueError("Invalid multiclass strategy. Choose 'ovr' or 'ovo'.")

    def predict(self, X, threshold=0, n_jobs=None):
        if self.multiclass_strategy == 'ovr':
            scores = np.zeros((len(X), len(self.models)))
            for idx, (svm_model, class_label) in enumerate(self.models):
                scores[:, idx] = svm_model.decision_function(X)
            
            # Choose the class with the highest score
            predictions = np.argmax(scores, axis=1)
            return np.array([self.models[i][1] for i in predictions])
        
        elif self.multiclass_strategy == 'ovo':
            votes = np.zeros((len(X), len(np.unique(self.Y))))
            for svm_model, class_1, class_2 in self.models:
                decision = svm_model.predict(X)
                for i, pred in enumerate(decision):
                    if pred == 1:
                        votes[i, class_1] += 1
                    else:
                        votes[i, class_2] += 1
            predictions=[]
            for i,vote in enumerate(votes):
                if np.argmax(vote)==0:
                    predictions.append( 0)
                if np.argmax(vote)==1:
                    predictions.append(1)
                if np.argmax(vote)==2:
                    predictions.append(-1)

            
            
            # Limitez les prédictions aux valeurs possibles : -1, 0, 1
            unique_classes = np.unique(self.Y)
            predictions = np.clip(predictions, unique_classes.min(), unique_classes.max())
            
            return predictions


    def decision_function(self, x):
        result = 0
        for i in range(self.m):
            result += self.alpha[i] * self.Y[i] * self.kernel_functions[self.kernel](self.X[i], x)
        return result + self.b

    def linear_kernel(self, x1, x2):
        return self.d * np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2, degree=2, c=1):
        return (np.dot(x1, x2) + c) ** degree

    def rbf_kernel(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

    def laplace_kernel(self, x1, x2, sigma=1):
        return np.exp(-np.linalg.norm(x1 - x2) / sigma)

    def matern_kernel(self, x1, x2, theta=1):
        distance = np.linalg.norm(x1 - x2)
        return (1 + (np.sqrt(5) * distance) / theta + (5 * distance ** 2) / (3 * theta ** 2)) * np.exp(-np.sqrt(5) * distance / theta)

    def distance_based_kernel(self, x1, x2):
        return self.d * (np.linalg.norm(x1) + np.linalg.norm(x2) - np.linalg.norm(x1 - x2))

    def cosine_kernel(self, x1, x2):
        return np.cos(np.dot(x1, x2))

    def sobolev_kernel(self, x1, x2):
        def k1(x):
            return x - 0.5
        def k2(x):
            return (k1(x) - 0.5) ** 2
        def k4(x):
            return (k1(x) ** 4 - k1(x) ** 2 / 2 + 7 / 240) / 24
        return k1(x1) * k1(x2) + k2(x1) * k2(x2) - k4(np.abs(x1 - x2))

    def product_operator_kernel(self, x1, x2):
        result = 1
        for xi, xj in zip(x1, x2):
            result *= self.rbf_kernel(xi, xj)
        return result
