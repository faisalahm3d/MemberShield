import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

from scripts.mia_attacks import membership_inference_attack
from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data_cifar10, \
    create_client_data_purchase100
from scripts.models import regularized_model_ch_minst, create_cnn, create_purchase_classifier, \
    regularized_model_ch_minst, regularized_purchase_classifier, vgg19_scratch
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, \
    visualize_decision_boundary
from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
exp_config = {
    'exp_name': 'anti-overfitting',
    'seed_value': 42,
    'learning_rate': 0.001,
    'momentum': 0.99,
    'n_round': 1,
    'n_clients': 5,
    'data_distribution': 'non-iid',
    'model': 'custom',
    'dataset': 'cifar10',
    'epochs': 1,
    'batch_size': 200,
    'n_attacks': 1,
    'server_type': 'honest_curious',
    'epsilon': 0.80,
    'optimizer': 'sgd',
    'label-type': 'hard',
    'loss': 'cce',
    'check': 'val_loss',
    'EarlyStop': 'No',
    'attack-type': 'both-threshold-knn-lr',
    'slice': 'entire-data',
    'class_label': 5

}
tf.random.set_seed(exp_config['seed_value'])
np.random.seed(exp_config['seed_value'])

file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
    exp_config['exp_name'],
    exp_config['dataset'],
    exp_config['model'],
    exp_config['label-type'],
    exp_config['loss'],
    exp_config['optimizer'],
    exp_config['learning_rate'],
    exp_config['momentum'],
    exp_config['check'],
    exp_config['EarlyStop'],
    exp_config['attack-type'],
    exp_config['slice']
)

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the root directory
root_directory = os.path.dirname(current_file_path)
# get the data directory
result_directory = os.path.join(root_directory, exp_config['dataset'] + '-' + exp_config['model'],
                                exp_config['exp_name'])
# summary result file path
file_path = os.path.join(result_directory, file_name)
file = open(file_path + '.txt', 'w')

# file = open(file_path, 'w')
print('Experiment details')
print('-------------------------------')
file.write('Experiment details\n')
file.write('-------------------------------\n')
for key, value in exp_config.items():
    print('{} : {}'.format(key, value))
    file.write('{} : {}\n'.format(key, value))
print('-------------------------------')
file.write('-------------------------------\n')

if exp_config['data_distribution'] == 'non-iid':
    if exp_config['dataset'] == 'cifar10':
        X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test, member_data, member_target, mem_entire_data, mem_entire_target = create_client_data_cifar10(
            exp_config['n_clients'], file)
    elif exp_config['dataset'] == 'purchase100':
        X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test, member_data, member_target, mem_entire_data, mem_entire_target = create_client_data_purchase100(
            exp_config['n_clients'], file)
    elif exp_config['dataset'] == 'ch-minst':
        X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test, member_data, member_target, mem_entire_data, mem_entire_target = create_client_data_ch_minst(
            exp_config['n_clients'], file)

else:
    X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test = client_data_two_class_each(
        exp_config['n_clients'], file)


def evaluate_client_model_accuracy_privacy_client_data(client_models, drop, reg, highest_score):
    summary_results = []
    for i in range(len(clients_data)):
        train_eva = client_models[i].evaluate(clients_data[i], clients_labels[i], verbose=0)
        val_eva = client_models[i].evaluate(clients_data_test[i], clients_labels_test[i], verbose=0)
        m_auc, m_adv = perform_mia(exp_config['n_attacks'], client_models[i], member_data[i], clients_data_test[i],
                                   member_target[i], clients_labels_test[i], exp_config['batch_size'])
        if m_auc > highest_score['AUC'][i]:
            highest_score['AUC'][i] = round(m_auc, 2)
        if m_adv > highest_score['ADV'][i]:
            highest_score['ADV'][i] = round(m_adv, 2)

        summary_results.append([
            i + 1,
            drop,
            reg,
            round(train_eva[0], 2),
            round(val_eva[0], 2),
            round(train_eva[1], 2),
            round(val_eva[1], 2),
            round(m_auc, 2),
            round(m_adv, 2)])
    heading_round = ['Client', 'Dropout', 'Reg', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on client models\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def evaluate_global_model_accuracy_privacy_client_data(global_model, drop, reg, highest_score_global):
    summary_results = []
    for i in range(len(clients_data)):
        train_eva = global_model.evaluate(clients_data[i], clients_labels[i], verbose=0)
        val_eva = global_model.evaluate(clients_data_test[i], clients_labels_test[i], verbose=0)
        m_auc, m_adv = perform_mia(exp_config['n_attacks'], global_model, member_data[i], clients_data_test[i],
                                   member_target[i], clients_labels_test[i], exp_config['batch_size'])
        if m_auc > highest_score_global['AUC'][i]:
            highest_score_global['AUC'][i] = round(m_auc, 2)
        if m_adv > highest_score_global['ADV'][i]:
            highest_score_global['ADV'][i] = round(m_adv, 2)

        summary_results.append([
            i + 1,
            drop,
            reg,
            round(train_eva[0], 2),
            round(val_eva[0], 2),
            round(train_eva[1], 2),
            round(val_eva[1], 2),
            round(m_auc, 2),
            round(m_adv, 2)])
    heading_round = ['Client', 'Dropout', 'Reg', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on global model\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def evaluate_global_model_accuracy_privacy_global_data(global_model, drop, reg):
    combined_member_data = np.concatenate(member_data, axis=0)
    combined_member_targets = np.concatenate(member_target, axis=0)
    summary_results = []
    train_eva = global_model.evaluate(X_train, y_train, verbose=0)
    val_eva = global_model.evaluate(X_test, y_test, verbose=0)
    m_auc, m_adv = perform_mia(exp_config['n_attacks'], global_model, combined_member_data, X_test,
                               combined_member_targets,
                               y_test,
                               exp_config['batch_size'])
    summary_results.append([
        drop,
        reg,
        round(train_eva[0], 2),
        round(val_eva[0], 2),
        round(train_eva[1], 2),
        round(val_eva[1], 2),
        round(m_auc, 2),
        round(m_adv, 2)])
    heading_round = ['drop', 'reg', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on global model considering whole training and testing data\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def perform_mia(n_attack, model, X_train, X_test, y_train, y_test, batch_size):
    aauc = []
    aadv = []
    for _ in range(n_attack):
        auc, adv = membership_inference_attack(model, X_train, X_test, y_train, y_test, batch_size, file)
        aauc.append(auc)
        aadv.append(adv)
    mauc = sum(aauc) / exp_config['n_attacks']
    madv = sum(aadv) / exp_config['n_attacks']
    return mauc, madv


def federated_training_anti_overfitting(global_model, *args):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=exp_config['learning_rate'])
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    highest_attack_performance_client = {
        'Client': [i + 1 for i in range(exp_config['n_clients'])],
        'AUC': [0.0 for i in range(exp_config['n_clients'])],
        'ADV': [0.0 for i in range(exp_config['n_clients'])]
    }
    highest_attack_performance_global = {
        'Client': [i + 1 for i in range(exp_config['n_clients'])],
        'AUC': [0.0 for i in range(exp_config['n_clients'])],
        'ADV': [0.0 for i in range(exp_config['n_clients'])]
    }
    training_time = 0
    for round_num in range(exp_config['n_round']):
        start_time = time.time()
        global_model_weight = global_model.get_weights()
        client_models = list()
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed
        for i in range(exp_config['n_clients']):
            start_time = time.time()
            client_model = tf.keras.models.clone_model(global_model)
            client_model.set_weights(global_model_weight)
            x, y = clients_data[i], clients_labels[i]
            client_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            client_model.fit(x,
                             y,
                             validation_data=(clients_data_test[i], clients_labels_test[i]),
                             epochs=exp_config['epochs'],
                             batch_size=exp_config['batch_size'],
                             verbose=0
                             )
            # here you can perform attach on each client after each iteration before aggregating them
            client_models.append(client_model)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed

            gen_error_acc, gen_error_loss = analyzer.analyze_visualize_model_characteristic(client_model, i, round_num,
                                                                                            loss, file_path+ 'drop{}_reg{}'.format(args[0], args[1]))
            # list_gen_error_acc[i + 1].append(gen_error_acc)
            # list_gen_error_loss[i + 1].append(gen_error_loss)

            # save_model(client_model,
            #            file_path + 'drop{}_reg{}_r{}_c{}.tf'.format(args[0], args[1], round_num + 1, i + 1))
            #
            # # analysis purpose
            # loss_train, loss_test, entropy_train, entropy_test = analysis_differences(
            #     client_model,
            #     member_data[i],
            #     clients_data_test[i],
            #     member_target[i],
            #     clients_labels_test[i],
            #     exp_config['batch_size']
            # )
            # draw_overlap_histogram(file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1, i + 1,
            #                        loss_train, loss_test, tag='losses')
            # draw_overlap_histogram(file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1, i + 1,
            #                        entropy_train, entropy_test, tag='entropies')
            #
            # visualize_decision_boundary(client_model.predict(member_data[i], batch_size=exp_config['batch_size']),
            #                             'member', file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1,
            #                             i + 1)
            # visualize_decision_boundary(client_model.predict(clients_data_test[i], batch_size=exp_config['batch_size']),
            #                             'non-member', file_path + 'drop{}_reg{}'.format(args[0], args[1]),
            #                             round_num + 1, i + 1)

            # perform_gradient_analysis(client_model,loss,x,y,clients_data_test[i], clients_labels_test[i])
        # Federated averaging (global aggregation)
        start_time = time.time()
        global_model = federated_averaging(global_model, client_models, num_clients=len(clients_data))
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed
        # g_weight = global_model.get_weights()
        global_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # loss_mem, loss_non, entropy_mem, entropy_non = analysis_differences(global_model, mem_entire_data,
        #                                                                     X_test,
        #                                                                     mem_entire_target,
        #                                                                     y_test,
        #                                                                     exp_config['batch_size'])
        # draw_overlap_histogram(file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1, None, loss_mem,
        #                        loss_non, tag='losses')
        # draw_overlap_histogram(file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1, None, entropy_mem,
        #                        entropy_non, tag='entropies')
        #
        # visualize_decision_boundary(global_model.predict(mem_entire_data, batch_size=exp_config['batch_size']),
        #                             'member', file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1)
        # visualize_decision_boundary(global_model.predict(X_test, batch_size=exp_config['batch_size']),
        #                             'non-member', file_path + 'drop{}_reg{}'.format(args[0], args[1]), round_num + 1)
        # save_model(global_model, file_path + 'drop{}_reg{}_r{}.tf'.format(args[0], args[1], round_num + 1))
        # ------------------------------------------------------------------------------
        file.write('Results after federated iteration # {}\n'.format(round_num + 1))
        file.write('==========================================================================\n')
        # mia on each individual clients for dp training
        evaluate_client_model_accuracy_privacy_client_data(client_models, args[0], args[1],
                                                           highest_attack_performance_client)
        # mia on global model after after aggregation
        evaluate_global_model_accuracy_privacy_client_data(global_model, args[0], args[1],
                                                           highest_attack_performance_global)
        # mia on global model with whole training and test data
        evaluate_global_model_accuracy_privacy_global_data(global_model, args[0], args[1])
    file.write('Summery Results: Client individual data\n')
    file.write('------------------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_client).to_string(index=False))
    file.write('\nSummery Results: Global data\n')
    file.write('---------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_global).to_string(index=False) + '\n')
    return global_model, global_model.evaluate(X_train, y_train, verbose=0), global_model.evaluate(X_test, y_test,
                                                                                                   verbose=0), training_time


settings = [
    (0.25, None),
    (0.50, None),
    (0.75, None),
    (None, 'l2'),
    (0.25, 'l2'),
    (0.50, 'l2'),
    (0.75, 'l2'),
]
fl_data = DataContainer(exp_config, file)
analyzer = BaseAnalyzer(exp_config, fl_data, file)
for drop, reg in settings:
    if exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'custom':
        model = create_cnn(X_train[0].shape, 10, drop, reg)
    elif exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'vgg19':
        model = vgg19_scratch(drop, reg)
    elif exp_config['dataset'] == 'purchase100' and exp_config['model'] == 'custom':
        model = regularized_purchase_classifier(100, drop, reg)
        # Display the model summary
        model.build((None, 600))
    elif exp_config['dataset'] == 'ch-minst' and exp_config['model'] == 'custom':
        model = regularized_model_ch_minst(drop, reg)
        # Display the model summary
        # model.build((None, 600))
    model, train_eva, val_eva, total_time = federated_training_anti_overfitting(model, drop, reg)
    file.write('---------------------------------------------------------\n')
    file.write('Time : {}\n'.format(total_time))
    file.write('Train Loss : {}\n'.format(round(train_eva[0], 2)))
    file.write('Train Acc: {}\n'.format(round(train_eva[1], 2)))
    file.write('Test Loss: {}\n'.format(round(val_eva[0], 2)))
    file.write('Test Acc: {}\n'.format(round(val_eva[1], 2)))
    file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
