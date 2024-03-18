import time
import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

from distiler import Distiller
from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data_purchase100, \
    create_client_data_cifar10
from scripts.mia_attacks import membership_inference_attack
from scripts.models import custom_model_ch_minst, create_model_softmax, create_purchase_classifier, vgg19_scratch
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, \
    visualize_decision_boundary, plot_hist, plot_dist, plot_ge_cdf, plot_prediction_probability, get_gradient_norm, \
    plot_generalization_error, class_wise_analysis

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# from gradient_analysis import perform_gradient_analysis

from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer

exp_config = {
    'exp_name': 'kd',
    'seed_value': 42,
    'learning_rate': 0.001,
    'momentum': 0.99,
    'n_round': 10,
    'n_clients': 5,
    'data_distribution': 'non-iid',
    'model': 'custom',
    'dataset': 'cifar10',
    'epochs': 50,
    'batch_size': 200,
    'n_attacks': 1,
    'distil_weight': 0.5,
    'temperature': 10,
    'server_type': 'honest_curious',
    'optimizer': 'sgd',
    'label-type': 'hard',
    'loss': 'cce',
    'check': 'val_loss',
    'EarlyStop': 'No',
    'attack-type': 'both-threshold-knn-lr',
    'slice': 'entire-data',
    'class_label': 5,
    'epsilon': 0.8
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
#file = open(file_path, 'w')
print('Experiment details')
print('-------------------------------')
file.write('Experiment details\n')
file.write('-------------------------------\n')
for key, value in exp_config.items():
    print('{} : {}'.format(key, value))
    file.write('{} : {}\n'.format(key, value))
print('-------------------------------')
file.write('-------------------------------\n')


def federated_training(global_model, *args):
    optimizer_adam = tf.keras.optimizers.legacy.Adam(learning_rate=exp_config['learning_rate'])
    optimizer_sgd = tf.keras.optimizers.legacy.SGD(
        learning_rate=exp_config['learning_rate'],
        momentum=exp_config['momentum'],
        nesterov=False,
        name='SGD',
    )
    optimizer = optimizer_sgd if exp_config['optimizer'] == 'sgd' else optimizer_adam
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
    list_gen_error_acc = {i: [] for i in range(1, exp_config['n_clients'] + 1)}
    list_gen_error_loss = {i: [] for i in range(1, exp_config['n_clients'] + 1)}
    entire_summary_results = {i: [] for i in range(1, exp_config['n_clients'] + 1)}
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
            client_model_student = tf.keras.models.clone_model(global_model)
            client_model.set_weights(global_model_weight)
            client_model_student.set_weights(global_model_weight)
            x, y = fl_data.clients_data[i], fl_data.clients_labels[i]
            client_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            client_model.fit(x,
                             y,
                             validation_data=(fl_data.clients_data_test[i], fl_data.clients_labels_test[i]),
                             epochs=exp_config['epochs'],
                             batch_size=exp_config['batch_size'],
                             verbose=0
                             )
            distiller = Distiller(student=client_model_student,
                                  teacher=client_model,
                                  optimizer=optimizer,
                                  metrics=[keras.metrics.CategoricalAccuracy()],
                                  student_loss_fn=loss,
                                  distillation_loss_fn=keras.losses.KLDivergence(),
                                  distil_weight=exp_config['distil_weight'],
                                  temperature=exp_config['temperature'])

            # Distill teacher to student
            distiller.fit(x, y, epochs=exp_config['epochs'], verbose=0)
            distil_client = distiller.get_student()
            distil_client.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            # save_model(client_model, file_path + 'r{}_c{}.tf'.format(round_num + 1, i + 1))

            # client_models.append(client_model)
            client_models.append(distil_client)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed
            gen_error_acc, gen_error_loss = analyzer.analyze_visualize_model_characteristic(client_model, i, round_num,
                                                                                            loss, file_path)

            # ---------------------------copied from no-defemse---------------------------
            # gradient_norm_members = [
            #     get_gradient_norm(distil_client, loss, fl_data.member_data[i][idx], fl_data.member_target[i][idx]) for
            #     idx in range(len(fl_data.member_data[i]))]
            # gradient_norm_non_members = [get_gradient_norm(distil_client, loss, fl_data.clients_data_test[i][idx],
            #                                                fl_data.clients_labels_test[i][idx]) for idx in
            #                              range(len(fl_data.clients_data_test[i]))]
            #
            #
            # plot_hist([gradient_norm_members, gradient_norm_non_members], ['member', 'non-member'], file_path,
            #           round_num + 1, i + 1, tag='gradient-norm')
            # plot_dist(gradient_norm_members, gradient_norm_non_members, '{}'.format(file_path),
            #           round_num + 1, i + 1, tag='Gradient norm')
            #
            # gen_error_acc, gen_error_loss, true_cls_prb_mem, true_cls_prob_non_mem, single_prob_mem, single_prob_non = class_wise_analysis(
            #     distil_client,
            #     fl_data.member_data[i],
            #     fl_data.clients_data_test[i],
            #     fl_data.member_target[i],
            #     fl_data.clients_labels_test[i],
            #     exp_config['batch_size'], class_label=exp_config['class_label'])
            #
            #
            #
            # list_gen_error_acc[i + 1].append(gen_error_acc)
            # list_gen_error_loss[i + 1].append(gen_error_loss)
            # plot_dist(true_cls_prb_mem, true_cls_prob_non_mem, '{}'.format(file_path), round_num + 1, i + 1,
            #           tag='Prediction probability')
            # plot_ge_cdf(gen_error_acc.values(), '{}_{}_{}_acc.png'.format(file_path, round_num + 1, i + 1))
            # plot_ge_cdf(gen_error_loss.values(), '{}_{}_{}_loss.png'.format(file_path, round_num + 1, i + 1))
            #
            # plot_prediction_probability(single_prob_mem, '{}_{}_{}_prob_mem'.format(file_path, round_num + 1, i + 1))
            # plot_prediction_probability(single_prob_non, '{}_{}_{}_prob_non'.format(file_path, round_num + 1, i + 1))
            #
            # # analysis purpose
            # loss_train, loss_test, entropy_train, entropy_test, m_entropy_train, m_entropy_test = analysis_differences(
            #     distil_client,
            #     fl_data.member_data[i],
            #     fl_data.clients_data_test[i],
            #     fl_data.member_target[i],
            #     fl_data.clients_labels_test[i],
            #     exp_config['batch_size'])
            #
            # plot_hist([loss_train, loss_test], ['member', 'non-member'], file_path, round_num + 1, i + 1, tag='Loss')
            # plot_hist([entropy_train, entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
            #           tag='Entropy')
            # plot_hist([m_entropy_train, m_entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
            #           tag='Modified entropy')
            #
            # draw_overlap_histogram(file_path, round_num + 1, i + 1, loss_train, loss_test, tag='Loss')
            # draw_overlap_histogram(file_path, round_num + 1, i + 1, entropy_train, entropy_test, tag='Entropy')
            #
            # visualize_decision_boundary(
            #     client_model.predict(fl_data.member_data[i], batch_size=exp_config['batch_size']),
            #     'member', file_path, round_num + 1, i + 1)
            # visualize_decision_boundary(
            #     client_model.predict(fl_data.clients_data_test[i], batch_size=exp_config['batch_size']),
            #     'non-member', file_path, round_num + 1, i + 1)

        start_time = time.time()
        # Federated averaging (global aggregation)
        global_model = federated_averaging(global_model, client_models, num_clients=len(fl_data.clients_data))
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed

        # g_weight = global_model.get_weights()
        global_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])



        # loss_mem, loss_non, entropy_mem, entropy_non, m_entropy_mem, m_entropy_non_mem = analysis_differences(
        #     global_model, fl_data.mem_entire_data,
        #     fl_data.X_test,
        #     fl_data.mem_entire_target,
        #     fl_data.y_test,
        #     exp_config['batch_size'])


        # ------------------------------------------------------------------------------
        file.write('Results after federated iteration # {}\n'.format(round_num + 1))
        file.write('==========================================================================\n')
        # mia on each individual clients for dp training
        analyzer.evaluate_client_model_accuracy_privacy_client_data(client_models, highest_attack_performance_client,
                                                                    entire_summary_results)
        # mia on global model after after aggregation
        analyzer.evaluate_global_model_accuracy_privacy_client_data(global_model, highest_attack_performance_global)
        # mia on global model with whole training and test data
        analyzer.evaluate_global_model_accuracy_privacy_global_data(global_model)
    file.write('Summery Results: Client individual data\n')
    file.write('------------------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_client).to_string(index=False))
    print()
    file.write('\nSummery Results: Global data\n')
    file.write('---------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_global).to_string(index=False) + '\n')

    # writer = pd.ExcelWriter(file_path + '.xlsx', engine='xlsxwriter')
    # for i in range(1, exp_config['n_clients'] + 1):
    #     df = pd.DataFrame(entire_summary_results[i])
    #     plot_generalization_error(df['train_acc'], df['test_acc'], df['train_loss'], df['test_loss'],
    #                               '{}_{}.png'.format(file_path, i))
    #     df.to_excel(writer, sheet_name='c{}'.format(i))
    #     df_acc = pd.DataFrame(list_gen_error_acc[i])
    #     df_acc.to_excel(writer, sheet_name='c{}_ge_acc'.format(i))
    #     df_loss = pd.DataFrame(list_gen_error_loss[i])
    #     df_loss.to_excel(writer, sheet_name='c{}_ge_loss'.format(i))
    # # Close the Pandas Excel writer and output the Excel file.
    # writer.close()

    return global_model, global_model.evaluate(fl_data.X_train, fl_data.y_train, verbose=0), global_model.evaluate(
        fl_data.X_test, fl_data.y_test,
        verbose=0), training_time


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
fl_data = DataContainer(exp_config, file)
analyzer = BaseAnalyzer(exp_config, fl_data, file)
model, train_eva, val_eva, total_time = federated_training(model)
file.write('---------------------------------------------------------\n')
file.write('Time : {}\n'.format(total_time))
file.write('Train Loss : {}\n'.format(train_eva[0]))
file.write('Test Loss: {}\n'.format(val_eva[0]))
file.write('Train Acc: {}\n'.format(train_eva[1]))
file.write('Test Acc: {}\n'.format(val_eva[1]))
file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
