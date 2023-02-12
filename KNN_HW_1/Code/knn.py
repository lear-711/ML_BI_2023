import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops = 0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = []
        
        for row_test in X:
            distances_test = []
            for row_train in self.train_X:
                distances_test.append(abs(row_train - row_test).sum())    
            distances.append(np.array(distances_test))
            
        return np.array(distances)


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances = []
        
        for row_test in X:
            distances_test = []
            row_test_dim = [row_test]*len(self.train_X)
            distances_test = np.absolute(row_test_dim - self.train_X).sum(axis=1)
            distances.append(distances_test)
            
        return np.array(distances)


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        return np.absolute(self.train_X - X[:,None]).sum(axis = 2)


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        ind_list = []
        value_list = []

        if self.k == 1:
            for i in range(n_test):
                min_ind = distances[i].argmin()
                prediction[i] = int(self.train_y[min_ind])
        elif self.k > 1:
            for i in range(n_test):
                min_values = sorted(distances[i])[:self.k]
                for each in min_values:
                    ind_list.append(np.where(distances[i] == each)[0][0])
                for ind in ind_list:
                    value_list.append(self.train_y[ind])
                unique, counts = np.unique(value_list, return_counts=True)
                unique_counts = {v:m for m,v in zip(unique, counts)}
                prediction[i] = unique_counts[max(unique_counts)]
    
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        ind_list = []
        value_list = []

        if self.k == 1:
            for i in range(n_test):
                min_ind = distances[i].argmin()
                prediction[i] = int(self.train_y[min_ind])
        elif self.k > 1:
            for i in range(n_test):
                min_values = sorted(distances[i])[:self.k]
                for each in min_values:
                    ind_list.append(np.where(distances[i] == each)[0][0])
                for ind in ind_list:
                    value_list.append(self.train_y[ind])
                unique, counts = np.unique(value_list, return_counts=True)
                unique_counts = {v:m for m,v in zip(unique, counts)}
                prediction[i] = unique_counts[max(unique_counts)]
    
        return prediction

