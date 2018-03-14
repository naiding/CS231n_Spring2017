import random
import numpy as np
from utils.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

from utils.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'utils/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

from utils.features import *

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])


################################################################################
# Use the validation set to tune the learning rate and regularization strength #
#                                       SVM                                    #
################################################################################

# from utils.classifiers.linear_classifier import LinearSVM
#
# learning_rates = [1e-9, 1e-8, 1e-7]
# regularization_strengths = [5e4, 5e5, 5e6]
#
# results = {}
# best_val = -1
# best_svm = None
#
# for lr in learning_rates:
#     for reg in regularization_strengths:
#         svm = LinearSVM()
#         _ = svm.train(X_train_feats, y_train, lr, reg, num_iters=1200, verbose=False)
#         y_train_pred = svm.predict(X_train_feats)
#         y_val_pred = svm.predict(X_val_feats)
#         y_train_accuracy = np.mean(y_train == y_train_pred)
#         y_val_accuracy = np.mean(y_val == y_val_pred)
#         results[(lr, reg)] = (y_train_accuracy, y_val_accuracy)
#         if y_val_accuracy > best_val:
#             best_val = y_val_accuracy
#             best_svm = svm
#
# # Print out results.
# for lr, reg in sorted(results):
#     train_accuracy, val_accuracy = results[(lr, reg)]
#     print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
#         lr, reg, train_accuracy, val_accuracy))
#
# print('best validation accuracy achieved during cross-validation: %f' % best_val)
#
# # Evaluate your trained SVM on the test set
# y_test_pred = best_svm.predict(X_test_feats)
# test_accuracy = np.mean(y_test == y_test_pred)
# print(test_accuracy)

################################################################################
# Use the validation set to tune the learning rate and regularization strength #
#                               two_layer_net                                  #
################################################################################

# from utils.classifiers.neural_net import TwoLayerNet
#
# input_dim = X_train_feats.shape[1]
# hidden_dim = 500
# num_classes = 10
#
# results = {}
# best_val = -1
# best_net = None  # store the best model into this
#
# learning_rates = [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
# regularization_strengths = [0.25, 0.5]
#
# for lr in learning_rates:
#     for r in regularization_strengths:
#         net = TwoLayerNet(input_dim, hidden_dim, num_classes)
#
#         _ = net.train(X_train_feats, y_train, X_val_feats, y_val, num_iters=1000, batch_size=200,
#                       learning_rate=lr, learning_rate_decay=0.95, reg=r, verbose=False)
#         y_train_pred = net.predict(X_train_feats)
#         y_val_pred = net.predict(X_val_feats)
#         y_train_accuracy = np.mean(y_train == y_train_pred)
#         y_val_accuracy = np.mean(y_val == y_val_pred)
#         results[(lr, r)] = (y_train_accuracy, y_val_accuracy)
#         if y_val_accuracy > best_val:
#             best_val = y_val_accuracy
#             best_net = net
#
# # Print out results.
# for lr, reg in sorted(results):
#     train_accuracy, val_accuracy = results[(lr, reg)]
#     print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
#         lr, reg, train_accuracy, val_accuracy))
#
# # Run your neural net classifier on the test set. You should be able to
# # get more than 55% accuracy.
#
# test_acc = (net.predict(X_test_feats) == y_test).mean()
# print(test_acc)
#
# # Run your neural net classifier on the test set. You should be able to
# # get more than 55% accuracy.
#
# test_acc = (best_net.predict(X_test_feats) == y_test).mean()
# print(test_acc)