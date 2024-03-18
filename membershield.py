import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from xlsxwriter import Workbook
import os
import pandas as pd

from scripts.datasetstf import create_client_data_ch_minst, create_client_data_cifar10, create_client_data_purchase100, \
    client_data_two_class_each
from scripts.mia_attacks import perform_mia
from scripts.models import custom_model_ch_minst, create_purchase_classifier, create_model_softmax, vgg19_scratch
from scripts.utility import entropy, get_top1, get_soft_labels, visualize_decision_boundary, analysis_differences, \
    draw_overlap_histogram, \
    perform_t_test, class_wise_analysis, get_gradient_norm, plot_hist, plot_ge_cdf, plot_dist, \
    plot_generalization_error, modified_entropy, plot_prediction_probability
from scripts.utility import federated_averaging, custom_cce, CustomEarlyStopping
from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# import custom modules

exp_config = {
    'exp_name': 'member-shield',
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
    'penalty': [0.0],
    'alpha': 'grid-search',
    'optimizer': 'sgd',
    'label-type': 'soft',
    'loss': 'el',
    'check': 'val_loss',
    'earlystop': 'Yes',
    'patience': 1,
    'attack-type': 'threshold-entropy-knn-lr',
    'slice': 'entire-data',
    'class_label': 5

}
tf.random.set_seed(exp_config['seed_value'])
np.random.seed(exp_config['seed_value'])

file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
    exp_config['exp_name'],
    exp_config['dataset'],
    exp_config['model'],
    exp_config['label-type'],
    exp_config['epsilon'],
    exp_config['loss'],
    exp_config['optimizer'],
    exp_config['learning_rate'],
    exp_config['momentum'],
    exp_config['alpha'],
    exp_config['check'],
    exp_config['earlystop'],
    exp_config['attack-type'],
    exp_config['slice'])

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


print('Experiment details')
print('-------------------------------')
file.write('Experiment details\n')
file.write('-------------------------------\n')
for key, value in exp_config.items():
    print('{} : {}'.format(key, value))
    file.write('{} : {}\n'.format(key, value))
print('-------------------------------')
file.write('-------------------------------\n')


def federated_training_member_shield(global_model, penalty):
    optimizer_adam = tf.keras.optimizers.legacy.Adam(learning_rate=exp_config['learning_rate'])
    optimizer_sgd = tf.keras.optimizers.legacy.SGD(
        learning_rate=exp_config['learning_rate'],
        momentum=exp_config['momentum'],
        nesterov=False,
        name='SGD',
    )
    optimizer = optimizer_sgd if exp_config['optimizer'] == 'sgd' else optimizer_adam
    loss_ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss_el = custom_cce(penalty)
    loss = loss_ce if exp_config['loss'] == 'cce' else loss_el
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
            # client_model.build(input_shape=(None, 32, 32, 3))
            client_model.set_weights(global_model_weight)
            x, y_hard, y_soft = fl_data.clients_data[i], fl_data.clients_labels[i], fl_data.clients_labels_modified[i]
            x_val = fl_data.clients_data_test[i]
            if exp_config['label-type'] == 'hard':
                y = y_hard
                y_val = fl_data.clients_labels_test[i]
            else:
                y = y_soft
                y_val = fl_data.clients_labels_test_modified[i]
            # early_stopping = EarlyStopping(monitor='val_loss', patience=exp_config['patience'],
            #                                restore_best_weights=True)

            # Usage
            client_model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
            initial_val_loss = client_model.evaluate(x_val, y_val, verbose=1)[0]
            custom_early_stopping = CustomEarlyStopping(initial_val_loss=initial_val_loss, monitor='val_loss',
                                                        patience=exp_config['patience'],
                                                        restore_best_weights=True)
            # client_model.summary()
            checkpoint_path = "training_checkpoint/cp_r{}_c{}.ckpt".format(round_num + 1, i + 1)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                     save_weights_only=True,
                                                                     save_best_only=True,
                                                                     monitor='val_loss',
                                                                     verbose=0)
            client_model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
            client_model.fit(
                x,
                y,
                validation_data=(x_val, y_val),
                epochs=exp_config['epochs'],
                batch_size=exp_config['batch_size'],
                callbacks=[custom_early_stopping, checkpoint_callback],
                verbose=0,
            )
            client_model.load_weights(checkpoint_path)
            client_models.append(client_model)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed

            client_model.compile(optimizer=optimizer, loss=loss_ce, metrics='accuracy')

            gen_error_acc, gen_error_loss = analyzer.analyze_visualize_model_characteristic(client_model, i, round_num,
                                                                                            loss, file_path)
            list_gen_error_acc[i + 1].append(gen_error_acc)
            list_gen_error_loss[i + 1].append(gen_error_loss)

        start_time = time.time()
        # Federated averaging (global aggregation)
        global_model = federated_averaging(global_model, client_models, num_clients=len(fl_data.clients_data))
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed

        global_model.compile(optimizer=optimizer, loss=loss_ce, metrics=['accuracy'])
        file.write('Results after federated iteration # {}\n'.format(round_num + 1))
        file.write('==========================================================================\n')
        # mia on each individual clients for dp training
        # print(en_list)
        # print(en_list_val)
        analyzer.evaluate_client_model_accuracy_privacy_client_data(client_models, highest_attack_performance_client,
                                                                    entire_summary_results)
        # mia on global model after after aggregation
        analyzer.evaluate_global_model_accuracy_privacy_client_data(global_model, highest_attack_performance_global)
        # mia on global model with whole training and test data
        analyzer.evaluate_global_model_accuracy_privacy_global_data(global_model)
    file.write('Summery Results: Client individual data\n')
    file.write('------------------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_client).to_string(index=False))
    file.write('\nSummery Results: Global data\n')
    file.write('---------------------------------\n')
    file.write(pd.DataFrame(highest_attack_performance_global).to_string(index=False) + '\n')

    writer = pd.ExcelWriter(file_path + '.xlsx', engine='xlsxwriter')
    for i in range(1, exp_config['n_clients'] + 1):
        df = pd.DataFrame(entire_summary_results[i])
        plot_generalization_error(df['train_acc'], df['test_acc'], df['train_loss'], df['test_loss'],
                                  '{}_{}_{}.png'.format(file_path, penalty, i))
        df.to_excel(writer, sheet_name='c{}'.format(i))
        df_acc = pd.DataFrame(list_gen_error_acc[i])
        df_acc.to_excel(writer, sheet_name='c{}_ge_acc'.format(i))
        df_loss = pd.DataFrame(list_gen_error_loss[i])
        df_loss.to_excel(writer, sheet_name='c{}_ge_loss'.format(i))
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()

    return global_model, global_model.evaluate(fl_data.X_train, fl_data.y_train, verbose=0), global_model.evaluate(
        fl_data.X_test, fl_data.y_test,
        verbose=0), training_time


for plty in exp_config['penalty']:
    if exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'custom':
        model = create_model_softmax()
    elif exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'vgg19':
        model = vgg19_scratch(10)
    elif exp_config['dataset'] == 'ch-minst' and exp_config['model'] == 'custom':
        model = custom_model_ch_minst()
    elif exp_config['dataset'] == 'purchase100' and exp_config['model'] == 'custom':
        model = create_purchase_classifier()
        # Display the model summary
        model.build((None, 600))
        # model.summary()
    fl_data = DataContainer(exp_config, file)
    analyzer = BaseAnalyzer(exp_config, fl_data, file)
    model_final, train_eva, val_eva, total_time = federated_training_member_shield(model, plty)
    file.write('---------------------------------------------------------\n')
    file.write('Time : {}\n'.format(total_time))
    file.write('Train Loss : {}\n'.format(train_eva[0]))
    file.write('Train Acc: {}\n'.format(train_eva[1]))
    file.write('Test Loss: {}\n'.format(val_eva[0]))
    file.write('Test Acc: {}\n'.format(val_eva[1]))
    file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
