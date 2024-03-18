import time
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from xlsxwriter import Workbook
import pandas as pd

from scripts.datasetstf import create_client_data_ch_minst, create_client_data, create_client_data_texas100, \
    client_data_two_class_each
from scripts.mia_attacks import perform_mia
from scripts.models import custom_model_ch_minst, create_purchase_classifier, create_model_softmax, vgg19_scratch
from scripts.utility import entropy, get_top1, get_soft_labels, visualize_decision_boundary, analysis_differences, \
    draw_overlap_histogram, \
    perform_t_test, class_wise_analysis, get_gradient_norm, plot_hist, plot_ge_cdf, plot_dist, \
    plot_generalization_error, modified_entropy, plot_prediction_probability
from scripts.utility import federated_averaging, custom_cce
from scripts.analysis import BaseAnalyzer
from scripts.adaptive_analysis import AdaptiveAnalyzer
from scripts.federated_data import DataContainer

# prob_logit = tf.constant(np.random.rand(1, 5))  # Replace with your actual data
# y_batch_train = tf.constant([0, 1, 0, 0, 0], shape=(1, 5), dtype=tf.float32)  # Replace with your actual data
# result = modified_entropy(prob_logit, y_batch_train)
# print(result)

print(f'\nTensorflow version = {tf.__version__}\n')
print(f'\n{tf.config.list_physical_devices("GPU")}\n')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# import custom modules

exp_config = {
    'exp_name': 'se-ce-.70-adaptive',
    'seed_value': 82,
    'learning_rate': 0.001,
    'momentum': 0.99,
    'n_round': 10,
    'n_clients': 5,
    'data_distribution': 'iid',
    'model': 'custom',
    'dataset': 'chifar10',
    'epochs': 50,
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
    'patience': 10,
    'attack-type': 'threshold-entropy-knn-lr',
    'slice': 'entire-data',
    'class_label': 5
}
tf.random.set_seed(exp_config['seed_value'])
np.random.seed(exp_config['seed_value'])

file_path = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.txt'.format(
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


# if exp_config['data_distribution'] == 'iid':
#     if exp_config['dataset'] == 'cifar10':
#         X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test, member_data, member_target, mem_entire_data, mem_entire_target = create_client_data(
#             exp_config['n_clients'], file)
#     elif exp_config['dataset'] == 'texas100':
#         X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test, member_data, member_target, mem_entire_data, mem_entire_target = create_client_data_texas100(
#             exp_config['n_clients'], file)
#     elif exp_config['dataset'] == 'ch-minst':
#         X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test, member_data, member_target, mem_entire_data, mem_entire_target = create_client_data_ch_minst(
#             exp_config['n_clients'], file)
# else:
#     X_train, y_train, X_test, y_test, clients_data, clients_labels, clients_data_test, clients_labels_test = client_data_two_class_each(
#         exp_config['n_clients'], file)
#
#
# def add_entropy(one_hot_label, epsilon):
#     soft_label = (1 - epsilon) * one_hot_label + epsilon / len(one_hot_label)
#     return soft_label / soft_label.sum()
#
#
# def generate_soft_labels(one_hot_labels):
#     soft = [add_entropy(label, exp_config['epsilon']) for label in one_hot_labels]
#     return np.array(soft)
#
#
# train_label_modified = generate_soft_labels(y_train.copy())
# test_label_modified = generate_soft_labels(y_test.copy())
#
# clients_labels_modified = [generate_soft_labels(client.copy()) for client in clients_labels]
#
# clients_labels_test_modified = [generate_soft_labels(client.copy()) for client in clients_labels_test]


# class custom_cce(tf.keras.losses.Loss):
#     def __init__(self, penalty):
#         super().__init__()
#         self.penalty = penalty
#
#     def call(self, y_true, y_pred):
#         y_pred = tf.nn.softmax(y_pred, axis=-1)
#         predicted_entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
#         loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#         #loss_fn = tf.keras.losses.KLDivergence()
#         loss_value = loss_fn(y_true, y_pred)
#         loss = loss_value - self.penalty * tf.reduce_mean(predicted_entropy)
#         return loss


# def evaluate_client_model_accuracy_privacy_client_data(client_models, highest_score, entire_summary_results):
#     summary_results = []
#     for i in range(len(clients_data)):
#         train_eva = client_models[i].evaluate(clients_data[i], clients_labels[i], verbose=1)
#         val_eva = client_models[i].evaluate(clients_data_test[i], clients_labels_test[i], verbose=1)
#         m_auc, m_adv = perform_mia(exp_config['n_attacks'], client_models[i], member_data[i], clients_data_test[i],
#                                    member_target[i], clients_labels_test[i], exp_config['batch_size'], file)
#         if m_auc > highest_score['AUC'][i]:
#             highest_score['AUC'][i] = round(m_auc, 2)
#         if m_adv > highest_score['ADV'][i]:
#             highest_score['ADV'][i] = round(m_adv, 2)
#         summary_results.append([
#             i + 1,
#             round(train_eva[0], 2),
#             round(val_eva[0], 2),
#             round(train_eva[1], 2),
#             round(val_eva[1], 2),
#             round(m_auc, 2),
#             round(m_adv, 2)])
#
#         temp_dict = {
#             'train_acc': round(train_eva[1], 2),
#             'test_acc': round(val_eva[1], 2),
#             'gen_error_acc': round(val_eva[1], 2) - round(train_eva[1], 2),
#             'train_loss': round(train_eva[0], 2),
#             'test_loss': round(val_eva[0], 2),
#             'gen_error_loss': round(val_eva[1], 2) - round(train_eva[1], 2),
#             'auc': round(m_auc, 2),
#             'adv': round(m_adv, 2)
#         }
#         entire_summary_results[i + 1].append(temp_dict)
#
#     heading_round = ['Client', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
#     df_result_round = pd.DataFrame(summary_results, columns=heading_round)
#     print(df_result_round.to_string(index=False))
#     print()
#     file.write('Attack on client models\n')
#     file.write('--------------------------------------------------------------------------\n')
#     file.write(df_result_round.to_string(index=False) + '\n')
#     file.write('--------------------------------------------------------------------------\n\n')


# def evaluate_global_model_accuracy_privacy_client_data(global_model, highest_score):
#     summary_results = []
#     for i in range(len(clients_data)):
#         train_eva = global_model.evaluate(clients_data[i], clients_labels[i], verbose=1)
#         val_eva = global_model.evaluate(clients_data_test[i], clients_labels_test[i], verbose=1)
#         m_auc, m_adv = perform_mia(exp_config['n_attacks'], global_model, member_data[i], clients_data_test[i],
#                                    member_target[i], clients_labels_test[i], exp_config['batch_size'], file)
#         if m_auc > highest_score['AUC'][i]:
#             highest_score['AUC'][i] = round(m_auc, 2)
#         if m_adv > highest_score['ADV'][i]:
#             highest_score['ADV'][i] = round(m_adv, 2)
#         summary_results.append([
#             i + 1,
#             round(train_eva[0], 2),
#             round(val_eva[0], 2),
#             round(train_eva[1], 2),
#             round(val_eva[1], 2),
#             round(m_auc, 2),
#             round(m_adv, 2)])
#     heading_round = ['Client', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
#     df_result_round = pd.DataFrame(summary_results, columns=heading_round)
#     print(df_result_round.to_string(index=False))
#     print()
#     file.write('Attack on global model\n')
#     file.write('--------------------------------------------------------------------------\n')
#     file.write(df_result_round.to_string(index=False) + '\n')
#     file.write('--------------------------------------------------------------------------\n\n')


# def evaluate_global_model_accuracy_privacy_global_data(global_model):
#     summary_results = []
#     train_eva = global_model.evaluate(X_train, y_train, verbose=1)
#     val_eva = global_model.evaluate(X_test, y_test, verbose=1)
#     m_auc, m_adv = perform_mia(exp_config['n_attacks'], global_model, mem_entire_data, X_test, mem_entire_target,
#                                y_test,
#                                exp_config['batch_size'], file)
#     summary_results.append([
#         round(train_eva[0], 2),
#         round(val_eva[0], 2),
#         round(train_eva[1], 2),
#         round(val_eva[1], 2),
#         round(m_auc, 2),
#         round(m_adv, 2)])
#     heading_round = ['Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
#     df_result_round = pd.DataFrame(summary_results, columns=heading_round)
#     print(df_result_round.to_string(index=False))
#     print()
#     file.write('Attack on global model considering whole training and testing data\n')
#     file.write('--------------------------------------------------------------------------\n')
#     file.write(df_result_round.to_string(index=False) + '\n')
#     file.write('--------------------------------------------------------------------------\n\n')


def federated_training_entropy_each_client(global_model, penalty):
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
            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=exp_config['patience'],
                                           restore_best_weights=True)
            # client_model.summary()
            checkpoint_path = "training_checkpoint/cp_r{}_c{}.ckpt".format(round_num + 1, i + 1)
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                     save_weights_only=True,
                                                                     save_best_only=True,
                                                                     monitor='val_loss',
                                                                     verbose=0)
            client_model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')
            history = client_model.fit(
                x,
                y,
                validation_data=(x_val, y_val),
                epochs=exp_config['epochs'],
                batch_size=exp_config['batch_size'],
                callbacks=[early_stopping, checkpoint_callback],
                verbose=0,
            )
            client_model.load_weights(checkpoint_path)
            client_models.append(client_model)
            end_time = time.time()
            time_elapsed = (end_time - start_time)
            training_time += time_elapsed

            # plot_generalization_error(history.history['accuracy'], history.history['val_accuracy'],
            #                           history.history['loss'], history.history[
            #                               'val_loss'], '{}{}.png'.format(penalty, file_path))

            client_model.compile(optimizer=optimizer, loss=loss_ce, metrics='accuracy')

            # client_model = tf.keras.models.load_model('best_model_round_{}_client_{}.tf'.format(round_num, i))
            gradient_norm_members = [
                get_gradient_norm(client_model, loss, fl_data.member_data[i][idx], fl_data.member_target[i][idx])
                for idx in range(len(fl_data.member_data[i]))]
            gradient_norm_non_members = [
                get_gradient_norm(client_model, loss, fl_data.clients_data_test[i][idx],
                                  fl_data.clients_labels_test[i][idx]) for idx in
                range(len(fl_data.clients_data_test[i]))]
            plot_hist([gradient_norm_members, gradient_norm_non_members], ['member', 'non-member'], file_path,
                      round_num + 1, i + 1, tag='gradient-norm')
            plot_dist(gradient_norm_members, gradient_norm_non_members, '{}{}'.format(penalty, file_path),
                      round_num + 1, i + 1, tag='Gradient norm')

            gen_error_acc, gen_error_loss, true_cls_prb_mem, true_cls_prob_non_mem, single_prob_mem, single_prob_non = class_wise_analysis(
                client_model,
                fl_data.member_data[i],
                fl_data.clients_data_test[
                    i],
                fl_data.member_target[
                    i],
                fl_data.clients_labels_test[
                    i],
                exp_config[
                    'batch_size'], class_label=exp_config['class_label'])
            list_gen_error_acc[i + 1].append(gen_error_acc)
            list_gen_error_loss[i + 1].append(gen_error_loss)
            plot_dist(true_cls_prb_mem, true_cls_prob_non_mem, '{}{}'.format(penalty, file_path), round_num + 1, i + 1,
                      tag='Prediction probability')
            plot_ge_cdf(gen_error_acc.values(), '{}_{}_{}_{}_acc.png'.format(penalty, file_path, round_num + 1, i + 1))
            plot_ge_cdf(gen_error_loss.values(),
                        '{}_{}_{}_{}_loss.png'.format(penalty, file_path, round_num + 1, i + 1))

            plot_prediction_probability(single_prob_mem,
                                        '{}_{}_{}_{}_prob_mem'.format(penalty, file_path, round_num + 1, i + 1))
            plot_prediction_probability(single_prob_non,
                                        '{}_{}_{}_{}_prob_non'.format(penalty, file_path, round_num + 1, i + 1))

            # analysis purpose
            loss_train, loss_test, entropy_train, entropy_test, m_entropy_train, m_entropy_test = analysis_differences(
                client_model, fl_data.member_data[i],
                fl_data.clients_data_test[i],
                fl_data.member_target[i],
                fl_data.clients_labels_test[i],
                exp_config['batch_size'])
            plot_hist([loss_train, loss_test], ['member', 'non-member'], file_path, round_num + 1, i + 1, tag='Loss')
            plot_hist([entropy_train, entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
                      tag='Entropy')
            plot_hist([m_entropy_train, m_entropy_test], ['member', 'non-member'], file_path, round_num + 1, i + 1,
                      tag='Modified entropy')
            # perform_t_test(loss_train, loss_test)
            # perform_t_test(entropy_train, entropy_test)
            draw_overlap_histogram('{}{}'.format(penalty, file_path), round_num + 1, i + 1, loss_train, loss_test,
                                   tag='Loss')
            draw_overlap_histogram('{}{}'.format(penalty, file_path), round_num + 1, i + 1, entropy_train, entropy_test,
                                   tag='Entropy')
            draw_overlap_histogram('{}{}'.format(penalty, file_path), round_num + 1, i + 1, m_entropy_train,
                                   m_entropy_test,
                                   tag='Modified entropy')
            visualize_decision_boundary(
                client_model.predict(fl_data.member_data[i], batch_size=exp_config['batch_size']),
                'member', '{}{}'.format(penalty, file_path), round_num + 1, i + 1)
            visualize_decision_boundary(
                client_model.predict(fl_data.clients_data_test[i], batch_size=exp_config['batch_size']),
                'non-member', '{}{}'.format(penalty, file_path), round_num + 1, i + 1)

            # save_model(client_model, '{}{}'.format(penalty, file_path) + 'r{}_c{}.tf'.format(round_num + 1, i + 1))

        start_time = time.time()
        # Federated averaging (global aggregation)
        global_model = federated_averaging(global_model, client_models, num_clients=len(fl_data.clients_data))
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        training_time += time_elapsed
        loss_mem, loss_non, entropy_mem, entropy_non, m_entropy_mem, m_entropy_non_mem = analysis_differences(
            global_model, fl_data.mem_entire_data,
            fl_data.X_test,
            fl_data.mem_entire_target,
            fl_data.y_test,
            exp_config['batch_size'])
        draw_overlap_histogram('{}{}'.format(penalty, file_path), round_num + 1, None, loss_mem, loss_non, tag='Loss')
        draw_overlap_histogram('{}{}'.format(penalty, file_path), round_num + 1, None, entropy_mem, entropy_non,
                               tag='Entropy')
        draw_overlap_histogram('{}{}'.format(penalty, file_path), round_num + 1, None, m_entropy_mem, m_entropy_non_mem,
                               tag='Modified entropy')
        visualize_decision_boundary(global_model.predict(fl_data.mem_entire_data, batch_size=exp_config['batch_size']),
                                    'member', '{}{}'.format(penalty, file_path), round_num + 1)
        visualize_decision_boundary(global_model.predict(fl_data.X_test, batch_size=exp_config['batch_size']),
                                    'non-member', '{}{}'.format(penalty, file_path), round_num + 1)
        # save_model(global_model, '{}{}'.format(penalty, file_path) + 'r{}.tf'.format(round_num + 1))
        # global_model = federated_average(global_model, client_models)

        # with keras.utils.custom_object_scope({'kl_loss': kl_loss}):
        global_model.compile(optimizer=optimizer, loss=loss_ce, metrics=['accuracy'])
        file.write('Results after federated iteration # {}\n'.format(round_num + 1))
        file.write('==========================================================================\n')

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
                                  '{}_{}_{}.png'.format(penalty, file_path, i))
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


# model = create_model_softmax(X_train[0].shape, 10)
# model = custom_model_ch_minst()

fl_data = DataContainer(exp_config, file)
analyzer = AdaptiveAnalyzer(exp_config, fl_data, file)
for plty in exp_config['penalty']:
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

    model_final, train_eva, val_eva, total_time = federated_training_entropy_each_client(model, plty)
    file.write('---------------------------------------------------------\n')
    file.write('Time : {}\n'.format(total_time))
    file.write('Train Loss : {}\n'.format(train_eva[0]))
    file.write('Train Acc: {}\n'.format(train_eva[1]))
    file.write('Test Loss: {}\n'.format(val_eva[0]))
    file.write('Test Acc: {}\n'.format(val_eva[1]))
    file.write('Time : {}\n'.format(total_time))
    file.write('---------------------------------------------------------\n')
file.close()
print('Training Complete')
