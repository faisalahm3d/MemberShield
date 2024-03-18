import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data, \
    create_client_data_texas100
from scripts.mia_attacks import membership_inference_attack
from scripts.models import custom_model_ch_minst, create_model_softmax, create_purchase_classifier, vgg19_scratch, \
    Attack, \
    inference_model
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, \
    visualize_decision_boundary, plot_hist
from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# from gradient_analysis import perform_gradient_analysis
# from custom_model_relax_loss import CustomModel

exp_config = {
    'exp_name': 'adv-reg',
    'seed_value': 82,
    'learning_rate': 0.001,
    'momentum': 0.99,
    'n_round': 10,
    'n_clients': 5,
    'epsilon': 0.7,
    'loss_threshold': 0.8,
    'data_distribution': 'iid',
    'model': 'vgg19',
    'dataset': 'cifar10',
    'num_class': 10,
    'epochs': 10,
    'batch_size': 200,
    'n_attacks': 1,
    'alpha': 3,
    'server_type': 'honest_curious',
    'optimizer': 'sgd',
    'label-type': 'hard',
    'loss': 'cce',
    'check': 'val_loss',
    'EarlyStop': 'No',
    'attack-type': 'both-threshold-knn-lr',
    'slice': 'entire-data'
}
tf.random.set_seed(exp_config['seed_value'])
np.random.seed(exp_config['seed_value'])

file_path = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(
    exp_config['exp_name'],
    exp_config['dataset'],
    exp_config['model'],
    exp_config['loss_threshold'],
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

file = open(file_path, 'w')
print('Experiment details')
print('-------------------------------')
file.write('Experiment details\n')
file.write('-------------------------------\n')
for key, value in exp_config.items():
    print('{} : {}'.format(key, value))
    file.write('{} : {}\n'.format(key, value))
print('-------------------------------')
file.write('-------------------------------\n')


def prepare_attack_input(y_true, y_pred):
    #return tf.concat([y_true, tf.cast(y_pred, tf.float64)], axis=1)
    return tf.concat([y_true, y_pred], axis=1)


def federated_training_with_relax_loss_paper(global_model, *args):
    # fl_data = DataContainer(exp_config, file)
    # analyzer = BaseAnalyzer(exp_config, fl_data, file)
    optimizer_adam = tf.keras.optimizers.legacy.Adam(learning_rate=exp_config['learning_rate'])
    optimizer_sgd = tf.keras.optimizers.legacy.SGD(
        learning_rate=exp_config['learning_rate'],
        momentum=exp_config['momentum'],
        nesterov=False,
        name='SGD',
    )
    optimizer = optimizer_sgd if exp_config['optimizer'] == 'sgd' else optimizer_adam
    loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

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
            client_model.set_weights(global_model_weight)

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (fl_data.clients_data[i], fl_data.clients_labels[i])).batch(exp_config['batch_size'])
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (fl_data.clients_data_test[i], fl_data.clients_labels_test[i])).batch(exp_config['batch_size'])
            # prepare attack model and data for training
            attack_model = Attack(exp_config['num_class'] * 2)
            attack_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # attack_model = inference_model(exp_config['num_class'], exp_config['num_class'], 1)

            # custom training function
            for epoch in range(exp_config['epochs']):
                total_loss = 0.0
                if epoch < 20:
                    # train unprotected model
                    @tf.function
                    def train_step(x_batch_train, y_batch_train):
                        with tf.GradientTape() as tape:
                            logit = client_model(x_batch_train, training=True)
                            prob_logit = tf.nn.softmax(logit, axis=-1)
                            loss = loss_fun(y_batch_train, prob_logit)
                        gradients = tape.gradient(loss, client_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, client_model.trainable_variables))
                        return loss

                    for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                        loss_value = train_step(x_batch_train, y_batch_train)
                        #print('Batch {}, Loss: {}'.format(epoch + 1, loss_value))
                        # total_loss += loss_value
                    #avg_loss = total_loss / batch
                    #print('Epoch {}, Loss: {}'.format(epoch + 1, avg_loss))
                else:
                    # train attack model
                    data_attack = tf.concat([fl_data.member_data[i], fl_data.clients_data_test[i]], axis=0)
                    attack_target = tf.concat(
                        [tf.ones(len(fl_data.member_data[i])), tf.zeros(len(fl_data.clients_data_test[i]))], axis=0)
                    attack_model_input1 = tf.concat([fl_data.member_target[i], fl_data.clients_labels_test[i]], axis=0)
                    attack_model_input2 = client_model.predict(data_attack, batch_size=exp_config['batch_size'])
                    # attack_model.fit([attack_model_input2, attack_model_input1], attack_target, epochs=50)
                    attack_input = prepare_attack_input(attack_model_input1, attack_model_input2)
                    attack_dataset = tf.data.Dataset.from_tensor_slices((attack_input, attack_target)).batch(
                        exp_config['batch_size']).shuffle(buffer_size=1024)

                    attack_model.fit(attack_input, attack_target, epochs=20, batch_size=exp_config['batch_size'],
                                     shuffle=True, verbose=0)

                    # @tf.function
                    # def train_step_attack(x, y):
                    #     with tf.GradientTape() as tape:
                    #         y_prob = attack_model(x, training=True)
                    #         loss_fun_attack = tf.keras.losses.BinaryCrossentropy()
                    #         loss = loss_fun_attack(y, y_prob)
                    #     gradients = tape.gradient(loss, attack_model.trainable_variables)
                    #     optimizer.apply_gradients(zip(gradients, attack_model.trainable_variables))
                    #     return loss
                    #
                    # for batch_attack, (x_batch_train_attack, y_batch_train_attack) in enumerate(attack_dataset):
                    #     attack_loss_value = train_step_attack(x_batch_train_attack, y_batch_train_attack)
                    #     total_loss_attack += attack_loss_value
                    # avg_loss_attack = total_loss / batch_attack
                    # print('Epoch {}, Loss: {}'.format(epoch + 1, avg_loss_attack))

                    # train protected model
                    @tf.function
                    def train_step_private(x_batch_train, y_batch_train):
                        with tf.GradientTape() as tape:
                            logit = client_model(x_batch_train, training=True)
                            prob_logit = tf.nn.softmax(logit, axis=-1)
                            attak_input = prepare_attack_input(y_batch_train, prob_logit)
                            inference_output = attack_model(attak_input)
                            # inference_output = attack_model([prob_logit, y_batch_train])
                            # loss = loss_fun(y_batch_train, prob_logit)
                            loss1 = loss_fun(y_batch_train, prob_logit)
                            # loss2 = tf.cast(exp_config['alpha'] * tf.math.log(tf.reduce_mean(inference_output)), tf.float64)
                            loss2 = exp_config['alpha'] * tf.math.log(tf.reduce_mean(inference_output))
                            # loss = loss_fun(prob_logit, y_batch_train) + exp_config['alpha'] * tf.math.log(
                            #     tf.reduce_mean(inference_output))
                            # loss = loss_fun(y_batch_train, prob_logit) + tf.cast(exp_config['alpha'] * tf.math.log(tf.reduce_mean(inference_output)), tf.float64)
                            loss = loss1 + loss2
                        gradients = tape.gradient(loss, client_model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, client_model.trainable_variables))
                        return loss

                    for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                        loss_value = train_step_private(x_batch_train, y_batch_train)
                        # total_loss += loss_value
                    # avg_loss = total_loss / batch
                    #print('Epoch {}, Loss: {}'.format(epoch + 1, loss_value))
                    # print("Time taken: %.2fs" % (time.time() - start_time))

            # x, y = clients_data[i], clients_labels[i]
            client_models.append(client_model)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed

            client_model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])
            # save_model(client_model, file_path + 'r{}_c{}.tf'.format(round_num + 1, i + 1))

            # analysis purpose
            # loss_train, loss_test, entropy_train, entropy_test = analysis_differences(client_model,
            #                                                                           fl_data.member_data[i],
            #                                                                           fl_data.clients_data_test[i],
            #                                                                           fl_data.member_target[i],
            #                                                                           fl_data.clients_labels_test[i],
            #                                                                           exp_config['batch_size'])
            #
            # plot_hist([loss_train, loss_test], ['member', 'non-member'], file_path, round_num + 1, i + 1, tag='losses')
            # plot_hist([entropy_train, entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
            #           tag='entropies')
            # draw_overlap_histogram(file_path, round_num + 1, i + 1, loss_train, loss_test, tag='losses')
            # draw_overlap_histogram(file_path, round_num + 1, i + 1, entropy_train, entropy_test, tag='entropies')
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

        global_model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])
        # save_model(global_model, file_path + 'r{}.tf'.format(round_num + 1))
        # ------------------------------------------------------------------------------
        file.write('Results after federated iteration # {}\n'.format(round_num + 1))
        file.write('==========================================================================\n')
        # mia on each individual clients for dp training
        analyzer.evaluate_client_model_accuracy_privacy_client_data(client_models, highest_attack_performance_client,entire_summary_results)
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
    return global_model, global_model.evaluate(fl_data.X_train, fl_data.y_train, verbose=0), global_model.evaluate(
        fl_data.X_test, fl_data.y_test,
        verbose=0), training_time


if exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'custom':
    model = create_model_softmax()
elif exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'vgg19':
    model = vgg19_scratch(10)
elif exp_config['dataset'] == 'ch-minst' and exp_config['model'] == 'custom':
    model = custom_model_ch_minst()
elif exp_config['dataset'] == 'purchase100' and exp_config['model'] == 'custom':
    model = create_purchase_classifier()
    model.build((None, 600))

fl_data = DataContainer(exp_config, file)
analyzer = BaseAnalyzer(exp_config, fl_data, file)
model, train_eva, val_eva, total_time = federated_training_with_relax_loss_paper(model)
file.write('---------------------------------------------------------\n')
file.write('Time : {}\n'.format(total_time))
file.write('Train Loss : {}\n'.format(round(train_eva[0], 2)))
file.write('Train Acc: {}\n'.format(round(train_eva[1], 2)))
file.write('Test Loss: {}\n'.format(round(val_eva[0], 2)))
file.write('Test Acc: {}\n'.format(round(val_eva[1], 2)))
file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
