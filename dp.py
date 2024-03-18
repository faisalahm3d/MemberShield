import time
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from scripts.mia_attacks import membership_inference_attack
from scripts.compute_dp_sgd_privacy_lib_copy import compute_dp_sgd_privacy_statement
from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data_cifar10, \
    create_client_data_purchase100
from scripts.models import custom_model_ch_minst, create_model_softmax, create_purchase_classifier, vgg19_scratch
from scripts.utility import calculate_noise
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, \
    visualize_decision_boundary
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
exp_config = {
    'exp_name': 'dp',
    'seed_value': 42,
    'learning_rate': 0.001,
    'momentum': 0.99,
    'n_round': 1,
    'n_clients': 5,
    'data_distribution': 'non-iid',
    'model': 'custom',
    'dataset': 'cifar10',
    'epochs': 1,
    'batch_size': 100,
    'n_attacks': 1,
    'microbatches': 1,
    'epsilons': [0.1, 0.5, 1, 2, 4, 8, 16, 100, 1000],
    'delta': 1e-6,
    'min_noise': 1e-10,
    'l2_norm_clip': 2.5,
    'server_type': 'honest_curious',
    'epsilon': 0.80,
    'optimizer': 'adam',
    'label-type': 'hard',
    'loss': 'cce',
    'check': 'val_loss',
    'EarlyStop': 'No',
    'attack-type': 'both-entropy-knn-lr',
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


def evaluate_client_model_accuracy_privacy_client_data(client_models, clients_noise_multiplier, clients_privacy_meet,
                                                       args, round_num, highest_score):
    # mia on each individual clients for dp training
    summary_results = []
    for i in range(len(clients_data)):
        train_eva = client_models[i].evaluate(clients_data[i], clients_labels[i], verbose=0)
        val_eva = client_models[i].evaluate(clients_data_test[i], clients_labels_test[i], verbose=0)
        m_auc, m_adv = membership_inference_attack(client_models[i], member_data[i], clients_data_test[i],
                                                   member_target[i], clients_labels_test[i], exp_config['batch_size'],
                                                   file)
        if m_auc > highest_score['AUC'][i]:
            highest_score['AUC'][i] = round(m_auc, 2)
        if m_adv > highest_score['ADV'][i]:
            highest_score['ADV'][i] = round(m_adv, 2)
        summary_results.append([
            i + 1,
            args[0],
            args[1],
            round(clients_noise_multiplier[i], 2),
            round(clients_privacy_meet[i], 2),
            round(train_eva[0], 2),
            round(val_eva[0], 2),
            round(train_eva[1], 2),
            round(val_eva[1], 2),
            round(m_auc, 2),
            round(m_adv, 2)])
    heading_round = ['Client', 'Target privacy', 'Delta', 'Noise Multiplier', 'Earned Privacy', 'Loss', 'Val loss',
                     'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Results after federated iteration # {}\n'.format(round_num + 1))
    file.write('==========================================================================\n')
    file.write('Attack on client models\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def evaluate_global_model_accuracy_privacy_client_data(global_model, clients_noise_multiplier, clients_privacy_meet,
                                                       args, highest_score):
    summary_results = []
    for i in range(len(clients_data)):
        train_eva = global_model.evaluate(clients_data[i], clients_labels[i], verbose=0)
        val_eva = global_model.evaluate(clients_data_test[i], clients_labels_test[i], verbose=0)
        m_auc, m_adv = membership_inference_attack(global_model, member_data[i], clients_data_test[i],
                                                   member_target[i], clients_labels_test[i],
                                                   exp_config['batch_size'], file)
        if m_auc > highest_score['AUC'][i]:
            highest_score['AUC'][i] = round(m_auc, 2)
        if m_adv > highest_score['ADV'][i]:
            highest_score['ADV'][i] = round(m_adv, 2)
        summary_results.append([
            i + 1,
            args[0],
            args[1],
            round(clients_noise_multiplier[i], 2),
            round(clients_privacy_meet[i], 2),
            round(train_eva[0], 2),
            round(val_eva[0], 2),
            round(train_eva[1], 2),
            round(val_eva[1], 2),
            round(m_auc, 2),
            round(m_adv, 2)])
    heading_round = ['Client', 'Target privacy', 'Delta', 'Noise Multiplier', 'Earned Privacy', 'Loss', 'Val loss',
                     'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on global model\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def evaluate_global_model_accuracy_privacy_global_data(global_model, noise_multiplier_global, earn_privacy_central,
                                                       args):
    summary_results = []
    train_eva = global_model.evaluate(X_train, y_train, verbose=0)
    val_eva = global_model.evaluate(X_test, y_test, verbose=0)
    m_auc, m_adv = membership_inference_attack(global_model, mem_entire_data, X_test, mem_entire_target, y_test,
                                               exp_config['batch_size'], file)
    summary_results.append([
        args[0],
        args[1],
        round(noise_multiplier_global, 2),
        round(earn_privacy_central, 2),
        round(train_eva[0], 2),
        round(val_eva[0], 2),
        round(train_eva[1], 2),
        round(val_eva[1], 2),
        round(m_auc, 2),
        round(m_adv, 2)])
    heading_round = ['Target privacy', 'Delta', 'Noise Multiplier', 'Earned Privacy', 'Loss', 'Val loss', 'Acc',
                     'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on global model considering whole training and testing data\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def federated_training_dp(global_model, *args):
    target_privacy = args[0]
    delta = args[1]

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

        clients_noise_multiplier = []
        clients_privacy_meet = []
        for i in range(exp_config['n_clients']):
            start_time = time.time()
            client_model = tf.keras.models.clone_model(global_model)
            client_model.set_weights(global_model_weight)
            x, y = clients_data[i], clients_labels[i]
            n = x.shape[0]
            noise_multiplier = calculate_noise(n, exp_config['batch_size'], target_privacy, exp_config['epochs'], delta,
                                               exp_config['min_noise'])
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=exp_config['l2_norm_clip'],
                noise_multiplier=noise_multiplier,
                num_microbatches=exp_config['microbatches'],
                learning_rate=exp_config['learning_rate']
            )
            # optimizer = DPKerasSGDOptimizer(
            #     l2_norm_clip=exp_config['l2_norm_clip'],
            #     noise_multiplier=noise_multiplier,
            #     num_microbatches=exp_config['microbatches'],
            #     learning_rate=exp_config['learning_rate'],
            #     momentum=exp_config['momentum'],
            #     nesterov=False,
            #     name='SGD',
            # )

            client_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            clients_noise_multiplier.append(noise_multiplier)
            earn_privacy, b, c, d = compute_dp_sgd_privacy_statement(number_of_examples=n,
                                                                     batch_size=exp_config['batch_size'],
                                                                     num_epochs=exp_config['epochs'],
                                                                     noise_multiplier=noise_multiplier,
                                                                     delta=exp_config['delta'],
                                                                     )
            clients_privacy_meet.append(earn_privacy)
            client_model.fit(x,
                             y,
                             validation_data=(clients_data_test[i], clients_labels_test[i]),
                             epochs=exp_config['epochs'],
                             batch_size=exp_config['batch_size'],
                             verbose=1
                             # callbacks=[callback]
                             )
            # here you can perform attach on each client after each iteration before aggregating them
            client_models.append(client_model)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed

            gen_error_acc, gen_error_loss = analyzer.analyze_visualize_model_characteristic(client_model, i, round_num,
                                                                                            loss,
                                                                                            file_path + 'epsilon{}_delta{}'.format(
                                                                                                args[0], args[1]))

            # save_model(client_model,
            #            file_path + 'epsilon{}_delta{}_r{}_c{}.tf'.format(args[0], args[1], round_num + 1, i + 1))
            # # analysis purpose
            # loss_train, loss_test, entropy_train, entropy_test = analysis_differences(
            #     client_model,
            #     member_data[i],
            #     clients_data_test[i],
            #     member_target[i],
            #     clients_labels_test[i],
            #     exp_config['batch_size']
            # )
            # draw_overlap_histogram(file_path + 'epsilon{}_delta{}'.format(args[0], args[1]), round_num + 1, i + 1,
            #                        loss_train, loss_test, tag='losses')
            # draw_overlap_histogram(file_path + 'epsilon{}_delta{}'.format(args[0], args[1]), round_num + 1, i + 1,
            #                        entropy_train, entropy_test, tag='entropies')
            # visualize_decision_boundary(client_model.predict(member_data[i], batch_size=exp_config['batch_size']),
            #                             'member', file_path + 'epsilon{}_delta{}'.format(args[0], args[1]),
            #                             round_num + 1, i + 1)
            # visualize_decision_boundary(client_model.predict(clients_data_test[i], batch_size=exp_config['batch_size']),
            #                             'non-member', file_path + 'epsilon{}_delta{}'.format(args[0], args[1]),
            #                             round_num + 1, i + 1)

        # Federated averaging (global aggregation)
        start_time = time.time()
        global_model = federated_averaging(global_model, client_models, num_clients=exp_config['n_clients'])
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed

        # This portion is necessary only for evaluation purpose
        noise_multiplier_global = calculate_noise(X_train.shape[0], exp_config['batch_size'], target_privacy,
                                                  exp_config['epochs'], delta, exp_config['min_noise'])
        earn_privacy_central, b_central, c_central, d_central = compute_dp_sgd_privacy_statement(
            number_of_examples=X_train.shape[0],
            batch_size=exp_config['batch_size'],
            num_epochs=exp_config['epochs'],
            noise_multiplier=noise_multiplier_global,
            delta=exp_config['delta'],
        )
        # optimizer = DPKerasSGDOptimizer(
        #     l2_norm_clip=exp_config['l2_norm_clip'],
        #     noise_multiplier=noise_multiplier_global,
        #     num_microbatches=exp_config['microbatches'],
        #     learning_rate=exp_config['learning_rate'],
        #     momentum=exp_config['momentum'],
        #     nesterov=False,
        #     name='SGD',
        # )
        global_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # draw_overlap_histogram(file_path + 'epsilon{}_delta{}'.format(args[0], args[1]), round_num + 1, None,
        #                        loss_train, loss_test, tag='losses')
        # draw_overlap_histogram(file_path + 'epsilon{}_delta{}'.format(args[0], args[1]), round_num + 1, None,
        #                        entropy_train, entropy_test, tag='entropies')
        #
        # visualize_decision_boundary(global_model.predict(mem_entire_data, batch_size=exp_config['batch_size']),
        #                             'member', file_path + 'epsilon{}_delta{}'.format(args[0], args[1]), round_num + 1)
        # visualize_decision_boundary(global_model.predict(X_test, batch_size=exp_config['batch_size']),
        #                             'non-member', file_path + 'epsilon{}_delta{}'.format(args[0], args[1]),
        #                             round_num + 1)
        # save_model(global_model, file_path + 'epsilon{}_delta{}_r{}.tf'.format(args[0], args[1], round_num + 1))
        # ------------------------------------------------------------------------------
        evaluate_client_model_accuracy_privacy_client_data(client_models, clients_noise_multiplier,
                                                           clients_privacy_meet, args, round_num,
                                                           highest_attack_performance_client)
        # mia on global model after after aggregation
        evaluate_global_model_accuracy_privacy_client_data(global_model, clients_noise_multiplier, clients_privacy_meet,
                                                           args, highest_attack_performance_global)
        # mia on global model with whole training and test data
        evaluate_global_model_accuracy_privacy_global_data(global_model, noise_multiplier_global, earn_privacy_central,
                                                           args)
    file.write('Summery Results: Client individual data\n')
    file.write('------------------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_client).to_string(index=False))
    print()
    file.write('\nSummery Results: Global data\n')
    file.write('---------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_global).to_string(index=False) + '\n')

    return global_model, global_model.evaluate(X_train, y_train, verbose=0), global_model.evaluate(X_test, y_test,
                                                                                                   verbose=0), training_time


fl_data = DataContainer(exp_config, file)
analyzer = BaseAnalyzer(exp_config, fl_data, file)
for e in exp_config['epsilons']:
    if exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'custom':
        model = create_model_softmax()
    elif exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'vgg19':
        model = vgg19_scratch(10)
    elif exp_config['dataset'] == 'ch-minst' and exp_config['model'] == 'custom':
        model = custom_model_ch_minst()
    elif exp_config['dataset'] == 'texas100' and exp_config['model'] == 'custom':
        model = create_purchase_classifier()
        # Display the model summary
        model.build((None, 600))
        # model.summary()
    model, train_eva, val_eva, total_time = federated_training_dp(model, e, exp_config['delta'])
    file.write('---------------------------------------------------------\n')
    file.write('Time : {}\n'.format(total_time))
    file.write('Train Loss : {}\n'.format(train_eva[0]))
    file.write('Train Acc: {}\n'.format(train_eva[1]))
    file.write('Test Loss: {}\n'.format(val_eva[0]))
    file.write('Test Acc: {}\n'.format(val_eva[1]))
    file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
