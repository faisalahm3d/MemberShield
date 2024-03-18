import numpy as np
from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data_cifar10, \
    create_client_data_purchase100, create_client_data_cifar100


class DataContainer:
    def __init__(self, exp_config, file):
        self.mem_entire_target = None
        self.mem_entire_data = None
        self.member_target = None
        self.member_data = None
        self.clients_labels_test = None
        self.clients_data_test = None
        self.clients_labels = None
        self.clients_data = None
        self.y_test = None
        self.X_test = None
        self.y_train = None
        self.X_train = None
        self.exp_config = exp_config
        self.file = file
        self.prepare_data()
        self.train_label_modified = self.generate_soft_labels(self.y_train.copy())
        self.test_label_modified = self.generate_soft_labels(self.y_test.copy())
        self.clients_labels_modified = [self.generate_soft_labels(client.copy()) for client in self.clients_labels]
        self.clients_labels_test_modified = [self.generate_soft_labels(client.copy()) for client in self.clients_labels_test]
        self.member_target_modified = [self.generate_soft_labels(member.copy()) for member in self.member_target]
        self.mem_entire_target_modified = self.generate_soft_labels(self.mem_entire_target.copy())

    def add_entropy(self, one_hot_label, epsilon):
        soft_label = (1 - epsilon) * one_hot_label + epsilon / len(one_hot_label)
        return soft_label / soft_label.sum()

    def generate_soft_labels(self, one_hot_labels):
        soft = [self.add_entropy(label, self.exp_config['epsilon']) for label in one_hot_labels]
        return np.array(soft)

    def prepare_data(self):
        if self.exp_config['data_distribution'] == 'non-iid':
            if self.exp_config['dataset'] == 'cifar10':
                self.X_train, self.y_train, self.X_test, self.y_test, self.clients_data, self.clients_labels, self.clients_data_test, self.clients_labels_test, self.member_data, self.member_target, self.mem_entire_data, self.mem_entire_target = create_client_data_cifar10(
                    self.exp_config['n_clients'], self.file)
            elif self.exp_config['dataset'] == 'cifar100':
                self.X_train, self.y_train, self.X_test, self.y_test, self.clients_data, self.clients_labels, self.clients_data_test, self.clients_labels_test, self.member_data, self.member_target, self.mem_entire_data, self.mem_entire_target = create_client_data_cifar100(
                    self.exp_config['n_clients'], self.file)
            elif self.exp_config['dataset'] == 'purchase100':
                self.X_train, self.y_train, self.X_test, self.y_test, self.clients_data, self.clients_labels, self.clients_data_test, self.clients_labels_test, self.member_data, self.member_target, self.mem_entire_data, self.mem_entire_target = create_client_data_purchase100(
                    self.exp_config['n_clients'], self.file)
            elif self.exp_config['dataset'] == 'ch-minst':
                self.X_train, self.y_train, self.X_test, self.y_test, self.clients_data, self.clients_labels, self.clients_data_test, self.clients_labels_test, self.member_data, self.member_target, self.mem_entire_data, self.mem_entire_target = create_client_data_ch_minst(
                    self.exp_config['n_clients'], self.file)
        else:
            self.X_train, self.y_train, self.X_test, self.y_test, self.clients_data, self.clients_labels, self.clients_data_test, self.clients_labels_test, self.member_data, self.member_target, self.mem_entire_data, self.mem_entire_target = client_data_two_class_each(
                self.exp_config['n_clients'], self.file)
