import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the root directory
root_directory = os.path.dirname(os.path.dirname(current_file_path))
# get the data directory
data_directory = os.path.join(root_directory, 'data')
ch_mnist_file_path = os.path.join(data_directory, 'hmnist_64_64_L.csv')
purchase100_file_path = os.path.join(data_directory, 'purchase100.npz')


def summerize_class_distribution(client_data, client_targets, file):
    n_clients = len(client_data)
    # Printing the sampled data statistics for each client
    str_rpt = ''
    for i in range(n_clients):
        print("Client {} data shape: {}".format(i + 1, client_data[i].shape))
        print(f"Client {i + 1} target shape:", client_targets[i].shape)

        file.write("Client {} data shape: {}\n".format(i + 1, client_data[i].shape))
        file.write("Client {} target shape: {}\n".format(i + 1, client_targets[i].shape))

    def class_distribution(targets):
        num_classes = targets.shape[1]
        class_counts = [int(tf.reduce_sum(targets[:, c])) for c in range(num_classes)]
        return class_counts

    # Printing the class-wise distribution for each client
    for i in range(n_clients):
        print(f"\n Client {i + 1} class-wise distribution:")
        file.write('\nClient {} class-wise distribution:'.format(i + 1))
        class_counts = class_distribution(client_targets[i])
        for class_idx, count in enumerate(class_counts):
            print(f"Class {class_idx}: {count}")
            file.write('Class {}: {}\n'.format(class_idx, count))
        file.write('\n')


def create_client_data_cifar10(n_clients, file, distributin_type='non-iid'):
    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train_), (X_test, y_test_) = cifar10.load_data()
    X_train, X_test = X_train.reshape((X_train.shape[0], 32, 32, 3)), X_test.reshape((X_test.shape[0], 32, 32, 3))
    X_train_flat, X_test_flat = X_train.reshape(-1, 32 * 32 * 3), X_test.reshape(-1, 32 * 32 * 3)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = np.eye(10)[y_train_.flatten()], np.eye(10)[y_test_.flatten()]

    n_datasample = len(X_train)
    client_proportions = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients = [int(client_proportions[i] * n_datasample) for i in range(n_clients)]

    permutation = np.random.permutation(len(X_train))
    shuffled_train_x = X_train[permutation]
    shuffled_train_y = y_train[permutation]

    client_data = [shuffled_train_x[start:start + size] for start, size in
                   zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]
    client_targets = [shuffled_train_y[start:start + size] for start, size in
                      zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]

    summerize_class_distribution(client_data, client_targets, file)

    n_datasample_test = len(X_test)
    # client_proportions_test = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients_test = [int(client_proportions[i] * n_datasample_test) for i in range(n_clients)]

    permutation_test = np.random.permutation(len(X_test))
    shuffled_test_x = X_test[permutation_test]
    shuffled_test_y = y_test[permutation_test]
    client_data_test = [shuffled_test_x[start:start + size] for start, size in
                        zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]
    client_targets_test = [shuffled_test_y[start:start + size] for start, size in
                           zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]

    summerize_class_distribution(client_data_test, client_targets_test, file)
    # Randomly select samples from the training dataset equal to the
    # number of test samples for each client to create member samples for the mia attack.
    member_data = list()
    member_target = list()
    for i in range(n_clients):
        _, X_member, _, y_member = train_test_split(client_data[i], client_targets[i],
                                                    test_size=n_sample_per_clients_test[i], stratify=client_targets[i])
        member_data.append(X_member)
        member_target.append(y_member)

    for i, mem in enumerate(member_data):
        assert len(mem), len(client_data_test[i])
    _, X_member_entire_data, _, y_member_entire_data = train_test_split(X_train, y_train, test_size=len(y_test),
                                                                        stratify=y_train)
    assert len(X_member_entire_data), len(y_test)

    return X_train, y_train, X_test, y_test, client_data, client_targets, client_data_test, client_targets_test, member_data, member_target, X_member_entire_data, y_member_entire_data


def create_client_data_cifar100(n_clients, file, distributin_type='non-iid'):
    cifar100 = tf.keras.datasets.cifar100
    (X_train, y_train_), (X_test, y_test_) = cifar100.load_data()
    X_train, X_test = X_train.reshape((X_train.shape[0], 32, 32, 3)), X_test.reshape((X_test.shape[0], 32, 32, 3))
    X_train_flat, X_test_flat = X_train.reshape(-1, 32 * 32 * 3), X_test.reshape(-1, 32 * 32 * 3)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = np.eye(100)[y_train_.flatten()], np.eye(100)[y_test_.flatten()]

    n_datasample = len(X_train)
    client_proportions = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients = [int(client_proportions[i] * n_datasample) for i in range(n_clients)]

    permutation = np.random.permutation(len(X_train))
    shuffled_train_x = X_train[permutation]
    shuffled_train_y = y_train[permutation]

    client_data = [shuffled_train_x[start:start + size] for start, size in
                   zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]
    client_targets = [shuffled_train_y[start:start + size] for start, size in
                      zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]

    summerize_class_distribution(client_data, client_targets, file)

    n_datasample_test = len(X_test)
    # client_proportions_test = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients_test = [int(client_proportions[i] * n_datasample_test) for i in range(n_clients)]

    permutation_test = np.random.permutation(len(X_test))
    shuffled_test_x = X_test[permutation_test]
    shuffled_test_y = y_test[permutation_test]
    client_data_test = [shuffled_test_x[start:start + size] for start, size in
                        zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]
    client_targets_test = [shuffled_test_y[start:start + size] for start, size in
                           zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]

    summerize_class_distribution(client_data_test, client_targets_test, file)
    # Randomly select samples from the training dataset equal to the
    # number of test samples for each client to create member samples for the mia attack.
    member_data = list()
    member_target = list()
    for i in range(n_clients):
        _, X_member, _, y_member = train_test_split(client_data[i], client_targets[i],
                                                    test_size=n_sample_per_clients_test[i], stratify=client_targets[i])
        member_data.append(X_member)
        member_target.append(y_member)

    for i, mem in enumerate(member_data):
        assert len(mem), len(client_data_test[i])
    _, X_member_entire_data, _, y_member_entire_data = train_test_split(X_train, y_train, test_size=len(y_test),
                                                                        stratify=y_train)
    assert len(X_member_entire_data), len(y_test)

    return X_train, y_train, X_test, y_test, client_data, client_targets, client_data_test, client_targets_test, member_data, member_target, X_member_entire_data, y_member_entire_data


def client_data_two_class_each(n_clients, file, distributin_type='non-iid'):
    cifar10 = tf.keras.datasets.cifar10
    cifar10_categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    (X_train, y_train_), (X_test, y_test_) = cifar10.load_data()
    X_train, X_test = X_train.reshape((X_train.shape[0], 32, 32, 3)), X_test.reshape((X_test.shape[0], 32, 32, 3))
    X_train_flat, X_test_flat = X_train.reshape(-1, 32 * 32 * 3), X_test.reshape(-1, 32 * 32 * 3)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = np.eye(10)[y_train_.flatten()], np.eye(10)[y_test_.flatten()]

    # Sort the dataset by class labels
    sort_indices = np.argsort(y_train_.flatten())
    train_data = X_train[sort_indices]
    train_labels = y_train[sort_indices]

    # Sort the dataset by class labels
    sort_indices_test = np.argsort(y_test_.flatten())
    test_data = X_test[sort_indices_test]
    test_labels = y_test[sort_indices_test]

    client_data = []
    client_labels = []

    start = 0
    step = 10000

    for i in range(n_clients):
        client_slice_x = train_data[start:start + step]
        client_slice_y = train_labels[start:start + step]
        client_data.append(client_slice_x)
        client_labels.append(client_slice_y)
        start = start + step

    client_data_test = []
    client_labels_test = []

    start = 0
    step = 2000

    for i in range(n_clients):
        client_slice_x = test_data[start:start + step]
        client_slice_y = test_labels[start:start + step]
        client_data_test.append(client_slice_x)
        client_labels_test.append(client_slice_y)
        start = start + step

    summerize_class_distribution(client_data, client_labels, file)
    summerize_class_distribution(client_data_test, client_labels_test, file)

    return X_train, y_train, X_test, y_test, client_data, client_labels, client_data_test, client_labels_test


def create_client_data_ch_minst(n_clients, file):
    # data = pd.read_csv("/home/AiTeam/FaisalA/membership-privacy/data/hmnist_64_64_L.csv")
    data = pd.read_csv(ch_mnist_file_path)
    Y = data["label"]
    data.drop(["label"], axis=1, inplace=True)
    X = data
    X = X.values.reshape(-1, 64, 64, 1)  # shaping for the Keras
    Y_ = Y.values
    Y_ = [y - 1 for y in Y]
    Y = to_categorical(Y_)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y_)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    n_datasample = len(X_train)
    client_proportions = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients = [int(client_proportions[i] * n_datasample) for i in range(n_clients)]

    permutation = np.random.permutation(len(X_train))
    shuffled_train_x = X_train[permutation]
    shuffled_train_y = Y_train[permutation]

    client_data = [shuffled_train_x[start:start + size] for start, size in
                   zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]
    client_targets = [shuffled_train_y[start:start + size] for start, size in
                      zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]

    summerize_class_distribution(client_data, client_targets, file)

    n_datasample_test = len(X_test)
    # client_proportions_test = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients_test = [int(client_proportions[i] * n_datasample_test) for i in range(n_clients)]

    permutation_test = np.random.permutation(len(X_test))
    shuffled_test_x = X_test[permutation_test]
    shuffled_test_y = Y_test[permutation_test]
    client_data_test = [shuffled_test_x[start:start + size] for start, size in
                        zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]
    client_targets_test = [shuffled_test_y[start:start + size] for start, size in
                           zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]

    summerize_class_distribution(client_data_test, client_targets_test, file)
    # Randomly select samples from the training dataset equal to the
    # number of test samples for each client to create member samples for the mia attack.
    member_data = list()
    member_target = list()
    for i in range(n_clients):
        _, X_member, _, y_member = train_test_split(client_data[i], client_targets[i],
                                                    test_size=n_sample_per_clients_test[i], stratify=client_targets[i])
        member_data.append(X_member)
        member_target.append(y_member)

    for i, mem in enumerate(member_data):
        assert len(mem), len(client_data_test[i])
    _, X_member_entire_data, _, y_member_entire_data = train_test_split(X_train, Y_train, test_size=len(Y_test),
                                                                        stratify=Y_train)
    assert len(X_member_entire_data), len(Y_test)

    return X_train, Y_train, X_test, Y_test, client_data, client_targets, client_data_test, client_targets_test, member_data, member_target, X_member_entire_data, y_member_entire_data


def create_client_data_purchase100(n_clients, file):
    # loaded_data = np.load('/home/AiTeam/FaisalA/membership-privacy/purchase100.npz')
    loaded_data = np.load(purchase100_file_path)

    # Access the arrays from the loaded data
    data = loaded_data['features']
    labels = loaded_data['labels']

    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
    # X_train, X_test = X_train / 255.0, X_test / 255.0

    n_datasample = len(X_train)
    client_proportions = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients = [int(client_proportions[i] * n_datasample) for i in range(n_clients)]

    permutation = np.random.permutation(len(X_train))
    shuffled_train_x = X_train[permutation]
    shuffled_train_y = Y_train[permutation]

    client_data = [shuffled_train_x[start:start + size] for start, size in
                   zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]
    client_targets = [shuffled_train_y[start:start + size] for start, size in
                      zip([0] + n_sample_per_clients[:-1], n_sample_per_clients)]

    summerize_class_distribution(client_data, client_targets, file)

    n_datasample_test = len(X_test)
    # client_proportions_test = np.random.dirichlet(tf.ones(n_clients))
    n_sample_per_clients_test = [int(client_proportions[i] * n_datasample_test) for i in range(n_clients)]

    permutation_test = np.random.permutation(len(X_test))
    shuffled_test_x = X_test[permutation_test]
    shuffled_test_y = Y_test[permutation_test]
    client_data_test = [shuffled_test_x[start:start + size] for start, size in
                        zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]
    client_targets_test = [shuffled_test_y[start:start + size] for start, size in
                           zip([0] + n_sample_per_clients_test[:-1], n_sample_per_clients_test)]

    summerize_class_distribution(client_data_test, client_targets_test, file)
    # Randomly select samples from the training dataset equal to the
    # number of test samples for each client to create member samples for the mia attack.
    member_data = list()
    member_target = list()
    for i in range(n_clients):
        _, X_member, _, y_member = train_test_split(client_data[i], client_targets[i],
                                                    test_size=n_sample_per_clients_test[i], stratify=client_targets[i])
        member_data.append(X_member)
        member_target.append(y_member)

    for i, mem in enumerate(member_data):
        assert len(mem), len(client_data_test[i])
    _, X_member_entire_data, _, y_member_entire_data = train_test_split(X_train, Y_train, test_size=len(Y_test),
                                                                        stratify=Y_train)
    assert len(X_member_entire_data), len(Y_test)

    return X_train, Y_train, X_test, Y_test, client_data, client_targets, client_data_test, client_targets_test, member_data, member_target, X_member_entire_data, y_member_entire_data
