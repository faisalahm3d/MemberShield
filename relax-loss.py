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
from scripts.models import custom_model_ch_minst, create_model_softmax, create_purchase_classifier, vgg19_scratch
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, \
    visualize_decision_boundary, plot_hist, get_gradient_norm, class_wise_analysis, plot_dist, \
    plot_generalization_error, plot_cdf, plot_ge_cdf, plot_prediction_probability
from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# from gradient_analysis import perform_gradient_analysis
# from custom_model_relax_loss import CustomModel

exp_config = {
    'exp_name': 'ralax-loss',
    'seed_value': 42,
    'learning_rate': 0.001,
    'momentum': 0.99,
    'n_round': 10,
    'n_clients': 5,
    'loss_threshold': 0.8,
    'data_distribution': 'iid',
    'model': 'custom',
    'dataset': 'cifar10',
    'num_class': 100,
    'epochs': 10,
    'epsilon': 0.4,
    'class_label': 50,
    'batch_size': 200,
    'n_attacks': 1,
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


def federated_training_with_relax_loss_paper(global_model, fl_data, analyzer, exp_config):
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
            client_model.set_weights(global_model_weight)
            x, y = fl_data.clients_data[i], fl_data.clients_labels[i]
            x_val, y_val = fl_data.clients_data_test[i], fl_data.clients_labels_test[i]
            train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(exp_config['batch_size'])
            test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(exp_config['batch_size'])

            # custom training function
            for epoch in range(exp_config['epochs']):
                total_loss = 0.0

                @tf.function
                def train_step(x_batch_train, y_batch_train):
                    with tf.GradientTape() as tape:
                        logit = client_model(x_batch_train, training=True)
                        prob_logit = tf.nn.softmax(logit, axis=-1)
                        loss_value = loss_fun(y_batch_train, prob_logit)
                        if loss_value >= exp_config['loss_threshold']:
                            loss = loss_value
                            # print('gradient descent')
                        else:
                            if epoch % 2 == 0:
                                # prob_train = tf.nn.softmax(logit, axis=-1)
                                # print('gradient ascent')
                                loss = tf.abs(loss_value - exp_config['loss_threshold'])
                                # print('gradient ascent')
                            else:
                                # Get the index of the true class
                                true_class_index = tf.argmax(y_batch_train, axis=-1)
                                # Get the softmax probability of the true class
                                true_class_prob = tf.gather(prob_logit, true_class_index, batch_dims=1)
                                # Calculate the remaining probability to be distributed
                                remaining_prob = 1.0 - true_class_prob
                                # Calculate the uniform probability for the other classes
                                uniform_prob = remaining_prob / (exp_config['num_class'] - 1)
                                # Create a tensor of uniform probabilities
                                uniform_probs = tf.ones_like(prob_logit) * tf.expand_dims(uniform_prob, axis=-1)
                                # Create a mask for the true class location
                                mask = tf.one_hot(true_class_index, depth=exp_config['num_class'])
                                # Create the new predictions
                                soft_targets = mask * tf.expand_dims(true_class_prob, axis=-1) + (
                                        1 - mask) * uniform_probs
                                # Calculate the loss on the soft targets
                                loss = loss_fun(y_batch_train, soft_targets)
                                # print('posterior flattening')
                        # loss_value = loss(y_batch_train, logits)
                    gradients = tape.gradient(loss, client_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, client_model.trainable_variables))
                    return loss

                for batch, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                    loss_value = train_step(x_batch_train, y_batch_train)
                    total_loss += loss_value
                avg_loss = total_loss / batch
                # print('Epoch {}, Loss: {}'.format(epoch + 1, avg_loss))
                # print("Time taken: %.2fs" % (time.time() - start_time))

            # x, y = clients_data[i], clients_labels[i]
            client_model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])
            client_models.append(client_model)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed

            # ---------------------------copied from no-defemse---------------------------
            gradient_norm_members = [
                get_gradient_norm(client_model, loss_fun, fl_data.member_data[i][idx], fl_data.member_target[i][idx])
                for
                idx in range(len(fl_data.member_data[i]))]
            gradient_norm_non_members = [get_gradient_norm(client_model, loss_fun, fl_data.clients_data_test[i][idx],
                                                           fl_data.clients_labels_test[i][idx]) for idx in
                                         range(len(fl_data.clients_data_test[i]))]

            # gradient_norm_members = get_gradient_norm_data(client_model, loss, fl_data.member_data[i], fl_data.member_target[i])
            # gradient_norm_non_members = get_gradient_norm_data(client_model, loss, fl_data.clients_data_test[i], fl_data.clients_labels_test[i])
            # plot_hist([gradient_norm_members, gradient_norm_non_members], ['member', 'non-member'], file_path,round_num + 1, i + 1, tag='gradient-norm')
            plot_dist(gradient_norm_members, gradient_norm_non_members, '{}'.format(file_path),
                      round_num + 1, i + 1, tag='Gradient norm')

            gen_error_acc, gen_error_loss, true_cls_prb_mem, true_cls_prob_non_mem, single_prob_mem, single_prob_non = class_wise_analysis(
                client_model,
                fl_data.member_data[i],
                fl_data.clients_data_test[i],
                fl_data.member_target[i],
                fl_data.clients_labels_test[i],
                exp_config['batch_size'], class_label=exp_config['class_label'])

            # get_gradient_norm(client_model, loss, fl_data.clients_data[i][0], fl_data.clients_labels[i][0])
            # class_wise_analysis(client_model, fl_data.member_data[i],
            #                     fl_data.clients_data_test[i],
            #                     fl_data.member_target[i],
            #                     fl_data.clients_labels_test[i],
            #                     exp_config['batch_size'])

            list_gen_error_acc[i + 1].append(gen_error_acc)
            list_gen_error_loss[i + 1].append(gen_error_loss)
            plot_dist(true_cls_prb_mem, true_cls_prob_non_mem, '{}'.format(file_path), round_num + 1, i + 1,
                      tag='Prediction probability')
            plot_ge_cdf(gen_error_acc.values(), '{}_{}_{}_acc.png'.format(file_path, round_num + 1, i + 1))
            plot_ge_cdf(gen_error_loss.values(), '{}_{}_{}_loss.png'.format(file_path, round_num + 1, i + 1))

            plot_prediction_probability(single_prob_mem, '{}_{}_{}_prob_mem'.format(file_path, round_num + 1, i + 1))
            plot_prediction_probability(single_prob_non, '{}_{}_{}_prob_non'.format(file_path, round_num + 1, i + 1))

            # analysis purpose
            loss_train, loss_test, entropy_train, entropy_test, m_entropy_train, m_entropy_test = analysis_differences(
                client_model,
                fl_data.member_data[i],
                fl_data.clients_data_test[i],
                fl_data.member_target[i],
                fl_data.clients_labels_test[i],
                exp_config['batch_size'])

            plot_hist([loss_train, loss_test], ['member', 'non-member'], file_path, round_num + 1, i + 1, tag='Loss')
            plot_hist([entropy_train, entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
                      tag='Entropy')
            plot_hist([m_entropy_train, m_entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
                      tag='Modified entropy')

            draw_overlap_histogram(file_path, round_num + 1, i + 1, loss_train, loss_test, tag='Loss')
            draw_overlap_histogram(file_path, round_num + 1, i + 1, entropy_train, entropy_test, tag='Entropy')

            visualize_decision_boundary(
                client_model.predict(fl_data.member_data[i], batch_size=exp_config['batch_size']),
                'member', file_path, round_num + 1, i + 1)
            visualize_decision_boundary(
                client_model.predict(fl_data.clients_data_test[i], batch_size=exp_config['batch_size']),
                'non-member', file_path, round_num + 1, i + 1)

            # analysis purpose
            # loss_train, loss_test, entropy_train, entropy_test = analysis_differences(distil_client, fl_data.member_data[i],
            #                                                                           fl_data.clients_data_test[i],
            #                                                                           fl_data.member_target[i],
            #                                                                           fl_data.clients_labels_test[i],
            #                                                                           exp_config['batch_size'])
            # plot_hist([loss_train, loss_test], ['member', 'non-member'], file_path, round_num + 1, i + 1, tag='losses')
            # plot_hist([entropy_train, entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
            #           tag='entropies')
            # draw_overlap_histogram(file_path, round_num + 1, i + 1, loss_train, loss_test, tag='losses')
            # draw_overlap_histogram(file_path, round_num + 1, i + 1, entropy_train, entropy_test, tag='entropies')
            # visualize_decision_boundary(client_model.predict(fl_data.member_data[i], batch_size=exp_config['batch_size']),
            #                             'member', file_path, round_num + 1, i + 1)
            # visualize_decision_boundary(client_model.predict(fl_data.clients_data_test[i], batch_size=exp_config['batch_size']),
            #                             'non-member', file_path, round_num + 1, i + 1)

        start_time = time.time()
        # Federated averaging (global aggregation)
        global_model = federated_averaging(global_model, client_models, num_clients=len(fl_data.clients_data))
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed

        # g_weight = global_model.get_weights()
        global_model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

        # visualize_decision_boundary(global_model.predict(fl_data.mem_entire_data, batch_size=exp_config['batch_size']),
        #                             'member', file_path, round_num + 1)
        # visualize_decision_boundary(global_model.predict(fl_data.X_test, batch_size=exp_config['batch_size']),
        #                             'non-member', file_path, round_num + 1)
        # save_model(global_model, file_path + 'r{}.tf'.format(round_num + 1))

        loss_mem, loss_non, entropy_mem, entropy_non, m_entropy_mem, m_entropy_non_mem = analysis_differences(
            global_model, fl_data.mem_entire_data,
            fl_data.X_test,
            fl_data.mem_entire_target,
            fl_data.y_test,
            exp_config['batch_size'])
        # draw_overlap_histogram(file_path, round_num + 1, None, loss_mem,loss_non, tag='Loss')
        # draw_overlap_histogram(file_path, round_num + 1, None, entropy_mem, entropy_non, tag='Entropy')
        # plot_hist([loss_mem, loss_non], ['member', 'non-member'], file_path, round_num + 1, i + 1, tag='Loss')
        # plot_hist([entropy_mem, entropy_non], ['member', 'non-member'], file_path, round_num + 1, i + 1,
        #           tag='Entropy')
        #
        # visualize_decision_boundary(global_model.predict(fl_data.mem_entire_data, batch_size=exp_config['batch_size']),
        #                             'member', file_path, round_num + 1)
        # visualize_decision_boundary(global_model.predict(fl_data.X_test, batch_size=exp_config['batch_size']),
        #                             'non-member', file_path, round_num + 1)

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

    writer = pd.ExcelWriter(file_path + '.xlsx', engine='xlsxwriter')
    for i in range(1, exp_config['n_clients'] + 1):
        df = pd.DataFrame(entire_summary_results[i])
        plot_generalization_error(df['train_acc'], df['test_acc'], df['train_loss'], df['test_loss'],
                                  '{}_{}.png'.format(file_path, i))
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



if exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'custom':
    model = create_model_softmax()
elif exp_config['dataset'] == 'cifar10' and exp_config['model'] == 'vgg19':
    model = vgg19_scratch(10)
elif exp_config['dataset'] == 'ch-minst' and exp_config['model'] == 'custom':
    model = custom_model_ch_minst()
elif exp_config['dataset'] == 'texas100' and exp_config['model'] == 'custom':
    model = create_purchase_classifier()
    # model = CustomModel(100, 0)
    # Display the model summary
    model.build((None, 600))
    # model.summary()
fl_data = DataContainer(exp_config, file)
analyzer = BaseAnalyzer(exp_config, fl_data, file)
model, train_eva, val_eva, total_time = federated_training_with_relax_loss_paper(model, fl_data, analyzer, exp_config)
file.write('---------------------------------------------------------\n')
file.write('Time : {}\n'.format(total_time))
file.write('Train Loss : {}\n'.format(round(train_eva[0], 2)))
file.write('Train Acc: {}\n'.format(round(train_eva[1], 2)))
file.write('Test Loss: {}\n'.format(round(val_eva[0], 2)))
file.write('Test Acc: {}\n'.format(round(val_eva[1], 2)))
file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
