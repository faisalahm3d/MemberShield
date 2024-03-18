import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model

from scripts.mia_attacks import membership_inference_attack
from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data, \
    create_client_data_texas100
from scripts.models import regularized_model_ch_minst, create_cnn, create_purchase_classifier, \
    regularized_model_ch_minst, regularized_purchase_classifier, vgg19_scratch
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, \
    visualize_decision_boundary


class AdaptiveAnalyzer:
    def __init__(self, exp_config, data, file):
        self.exp_config = exp_config
        self.data = data
        self.file = file

    def evaluate_client_model_accuracy_privacy_client_data(self, client_models, highest_score, entire_summary_results):
        summary_results = []
        for i in range(len(self.data.clients_data)):
            train_eva = client_models[i].evaluate(self.data.clients_data[i], self.data.clients_labels[i], verbose=0)
            val_eva = client_models[i].evaluate(self.data.clients_data_test[i], self.data.clients_labels_test[i],
                                                verbose=0)
            m_auc, m_adv = self.perform_mia(self.exp_config['n_attacks'], client_models[i], self.data.member_data[i],
                                            self.data.clients_data_test[i],
                                            self.data.member_target_modified[i], self.data.clients_labels_test_modified[i],
                                            self.exp_config['batch_size'])
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
            temp_dict = {
                'train_acc': round(train_eva[1], 2),
                'test_acc': round(val_eva[1], 2),
                'gen_error_acc': round(val_eva[1], 2) - round(train_eva[1], 2),
                'train_loss': round(train_eva[0], 2),
                'test_loss': round(val_eva[0], 2),
                'gen_error_loss': round(val_eva[1], 2) - round(train_eva[1], 2),
                'auc': round(m_auc, 2),
                'adv': round(m_adv, 2)
            }
            entire_summary_results[i + 1].append(temp_dict)
        heading_round = ['Client', 'Loss', 'Val loss', 'Acc', 'Val_Acc', 'AUC', 'Adv']
        df_result_round = pd.DataFrame(summary_results, columns=heading_round)
        print(df_result_round.to_string(index=False))
        print()
        self.file.write('Attack on client models\n')
        self.file.write('--------------------------------------------------------------------------\n')
        self.file.write(df_result_round.to_string(index=False) + '\n')
        self.file.write('--------------------------------------------------------------------------\n\n')

    # def evaluate_client_model_accuracy_privacy_client_data(self, client_models, highest_score, entire_summary_results):
    #     summary_results = []
    #     for i in range(len(self.data.clients_data)):
    #         train_eva = client_models[i].evaluate(self.data.clients_data[i], self.data.clients_labels[i], verbose=0)
    #         val_eva = client_models[i].evaluate(self.data.clients_data_test[i], self.data.clients_labels_test[i], verbose=0)
    #         m_auc, m_adv = self.perform_mia(self.exp_config['n_attacks'], client_models[i], self.data.member_data[i], self.data.clients_data_test[i],
    #                                    self.data.member_target[i], self.data.clients_labels_test[i], self.exp_config['batch_size'])
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
    #     self.file.write('Attack on client models\n')
    #     self.file.write('--------------------------------------------------------------------------\n')
    #     self.file.write(df_result_round.to_string(index=False) + '\n')
    #     self.file.write('--------------------------------------------------------------------------\n\n')

    def evaluate_global_model_accuracy_privacy_client_data(self, global_model, highest_score_global):
        summary_results = []
        for i in range(len(self.data.clients_data)):
            train_eva = global_model.evaluate(self.data.clients_data[i], self.data.clients_labels[i], verbose=0)
            val_eva = global_model.evaluate(self.data.clients_data_test[i], self.data.clients_labels_test[i], verbose=0)
            m_auc, m_adv = self.perform_mia(self.exp_config['n_attacks'], global_model, self.data.member_data[i],
                                            self.data.clients_data_test[i],
                                            self.data.member_target_modified[i],
                                            self.data.clients_labels_test_modified[i],
                                            self.exp_config['batch_size'])
            if m_auc > highest_score_global['AUC'][i]:
                highest_score_global['AUC'][i] = round(m_auc, 2)
            if m_adv > highest_score_global['ADV'][i]:
                highest_score_global['ADV'][i] = round(m_adv, 2)

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
        self.file.write('Attack on global model\n')
        self.file.write('--------------------------------------------------------------------------\n')
        self.file.write(df_result_round.to_string(index=False) + '\n')
        self.file.write('--------------------------------------------------------------------------\n\n')

    def evaluate_global_model_accuracy_privacy_global_data(self, global_model):
        combined_member_data = np.concatenate(self.data.member_data, axis=0)
        combined_member_targets = np.concatenate(self.data.member_target_modified, axis=0)
        summary_results = []
        train_eva = global_model.evaluate(self.data.X_train, self.data.y_train, verbose=0)
        val_eva = global_model.evaluate(self.data.X_test, self.data.y_test, verbose=0)
        m_auc, m_adv = self.perform_mia(self.exp_config['n_attacks'],
                                        global_model, combined_member_data,
                                        self.data.X_test,
                                        combined_member_targets,
                                        self.data.test_label_modified,
                                        self.exp_config['batch_size'])
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
        self.file.write('Attack on global model considering whole training and testing data\n')
        self.file.write('--------------------------------------------------------------------------\n')
        self.file.write(df_result_round.to_string(index=False) + '\n')
        self.file.write('--------------------------------------------------------------------------\n\n')

    def perform_mia(self, n_attack, model, X_mem, X_nonmember, y_mem, y_nonmember, batch_size):
        aauc = []
        aadv = []
        for _ in range(n_attack):
            auc, adv = membership_inference_attack(model, X_mem, X_nonmember, y_mem, y_nonmember, batch_size, self.file)
            aauc.append(auc)
            aadv.append(adv)
        mauc = sum(aauc) / self.exp_config['n_attacks']
        madv = sum(aadv) / self.exp_config['n_attacks']
        return mauc, madv
