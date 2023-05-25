import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        if (y_pred[i] == 1) and (y_true[i] == 1):
            tp += 1
        elif (y_pred[i] == 1) and (y_true[i] == 0):
            fp += 1
        elif (y_pred[i] == 0) and (y_true[i] == 1):
            fn += 1
        elif (y_pred[i] == 0) and (y_true[i] == 0):
            tn += 1
        
    if ((tp + fp) != 0):
        precision = tp/(tp + fp)
    else:
        precision = 0
        print('Division by zero!')
        
    if ((tp + fn) != 0):
        recall = tp/(tp + fn)
    else:
        recall = 0
        print('Division by zero!')
        
    if ((tp + tn + fp + fn) != 0):
        accuracy = (tp + tn)/(tp + tn + fp + fn)
    else:
        accuracy = 0
        print('Division by zero!')
        
    if (( 2*tp + fp + fn) != 0):
        f1_score = 2*tp/( 2*tp + fp + fn)
    else:
        f1_score = 0
        print('Division by zero!')
        
        
    return precision, recall, accuracy, f1_score
    
    

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        if (y_pred[i] == 1) and (y_true[i] == 1):
            tp += 1
        elif (y_pred[i] == 1) and (y_true[i] == 0):
            fp += 1
        elif (y_pred[i] == 0) and (y_true[i] == 1):
            fn += 1
        elif (y_pred[i] == 0) and (y_true[i] == 0):
            tn += 1
        
    if ((tp + tn + fp + fn) != 0):
        accuracy = (tp + tn)/(tp + tn + fp + fn)
    else:
        accuracy = 0
        print('Division by zero!')
        
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    ver = 0
    nis = 0
    mean_true = np.mean(y_true)

    for i in range(len(y_pred)):
        ver += np.power((y_true[i] - y_pred[i]), 2)
        nis += np.power((y_true[i] - mean_true), 2)

    r_2 = 1 - ver/nis
    return r_2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    
    ver = 0
    n = len(y_true)
    
    for i in range(len(y_pred)):
        ver += np.power((y_true[i] - y_pred[i]), 2)
    
    mse = ver/n
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    ver = 0
    n = len(y_true)
    
    for i in range(len(y_pred)):
        ver += abs(y_true[i] - y_pred[i])
    
    mae = ver/n
    return mae
    