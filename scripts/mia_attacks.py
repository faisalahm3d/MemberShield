import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import numpy as np
import pandas as pd
from scripts.utility import draw_histogram, draw_overlap_histogram


def membership_inference_attack(model, X_train, X_test, y_train, y_test, batch_size, file_path=None):
    logit_train = model.predict(X_train, batch_size=batch_size, verbose=0)
    logit_test = model.predict(X_test, batch_size=batch_size, verbose=0)
    prob_train = tf.nn.softmax(logit_train, axis=-1)
    prob_test = tf.nn.softmax(logit_test)
    # entropy_train = -tf.reduce_sum(logits_train * tf.math.log(logits_train), axis=-1)
    # entropy_test = -tf.reduce_sum(logits_test * tf.math.log(logits_test), axis=-1)
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    loss_train = cce(constant(y_train), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test), constant(prob_test), from_logits=False).numpy()

    labels_train = np.argmax(y_train, axis=1)
    labels_test = np.argmax(y_test, axis=1)

    input = AttackInputData(
        logits_train=logit_train,
        logits_test=logit_test,
        loss_train=loss_train,
        loss_test=loss_test,
        labels_train=labels_train,
        labels_test=labels_test
    )
    # Run several attacks for different data slices
    attacks_result = mia.run_attacks(input,
                                     SlicingSpec(
                                         entire_dataset=True,
                                         # by_class=True,
                                         # by_classification_correctness=True
                                     ),
                                     attack_types=[
                                         AttackType.THRESHOLD_ENTROPY_ATTACK,
                                         AttackType.THRESHOLD_ATTACK,
                                         AttackType.LOGISTIC_REGRESSION,
                                         # AttackType.MULTI_LAYERED_PERCEPTRON,
                                         # AttackType.RANDOM_FOREST,
                                         AttackType.K_NEAREST_NEIGHBORS
                                     ])
    # print(attacks_result.summary(by_slices=True))
    file_path.write(attacks_result.summary(by_slices=True))
    # file_path.write('\n')
    return attacks_result.get_result_with_max_auc().get_auc(), attacks_result.get_result_with_max_attacker_advantage().get_attacker_advantage()


def perform_mia(attack_number, model, X_train, X_test, y_train, y_test, batch_size, file):
    aauc = []
    aadv = []
    for _ in range(attack_number):
        auc, adv = membership_inference_attack(model, X_train, X_test, y_train, y_test, batch_size, file)
        aauc.append(auc)
        aadv.append(adv)
    mauc = sum(aauc) / attack_number
    madv = sum(aadv) / attack_number
    return mauc, madv


def evaluate_client_model_accuracy_privacy_client_data(
        client_models,
        clients_data,
        clients_labels,
        clients_data_test,
        clients_labels_test,
        exp_config,
        member_data,
        member_target,
        highest_score,
        file):
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
            round(train_eva[0], 2),
            round(val_eva[0], 2),
            round(train_eva[1], 2),
            round(val_eva[1], 2),
            round(m_auc, 2),
            round(m_adv, 2)])
    heading_round = ['Client', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on client models\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def evaluate_global_model_accuracy_privacy_client_data(global_model,
                                                       clients_data,
                                                       clients_labels,
                                                       clients_data_test,
                                                       clients_labels_test,
                                                       exp_config,
                                                       member_data,
                                                       member_target,
                                                       highest_score,
                                                       file):
    summary_results = []
    for i in range(len(clients_data)):
        train_eva = global_model.evaluate(clients_data[i], clients_labels[i], verbose=0)
        val_eva = global_model.evaluate(clients_data_test[i], clients_labels_test[i], verbose=0)
        m_auc, m_adv = perform_mia(exp_config['n_attacks'], global_model, member_data[i], clients_data_test[i],
                                   member_target[i], clients_labels_test[i], exp_config['batch_size'])
        if m_auc > highest_score['AUC'][i]:
            highest_score['AUC'][i] = round(m_auc, 2)
        if m_adv > highest_score['ADV'][i]:
            highest_score['ADV'][i] = round(m_adv, 2)
        summary_results.append([
            i + 1,
            round(train_eva[0], 2),
            round(val_eva[0], 2),
            round(train_eva[1], 2),
            round(val_eva[1], 2),
            round(m_auc, 2),
            round(m_adv, 2)])
    heading_round = ['Client', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on global model\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')


def evaluate_global_model_accuracy_privacy_global_data(
        global_model,
        X_train,
        y_train,
        X_test,
        y_test,
        mem_entire_data,
        mem_entire_target,
        exp_config,
        file
):
    summary_results = []
    train_eva = global_model.evaluate(X_train, y_train, verbose=0)
    val_eva = global_model.evaluate(X_test, y_test, verbose=0)
    m_auc, m_adv = perform_mia(exp_config['n_attacks'], global_model, mem_entire_data, X_test, mem_entire_target,
                               y_test,
                               exp_config['batch_size'])
    summary_results.append([
        round(train_eva[0], 2),
        round(val_eva[0], 2),
        round(train_eva[1], 2),
        round(val_eva[1], 2),
        round(m_auc, 2),
        round(m_adv, 2)])
    heading_round = ['Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
    df_result_round = pd.DataFrame(summary_results, columns=heading_round)
    print(df_result_round.to_string(index=False))
    print()
    file.write('Attack on global model considering whole training and testing data\n')
    file.write('--------------------------------------------------------------------------\n')
    file.write(df_result_round.to_string(index=False) + '\n')
    file.write('--------------------------------------------------------------------------\n\n')
