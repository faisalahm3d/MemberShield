import time
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import numpy as np
import pandas as pd
import tensorflow as tf

from scripts.datasetstf import client_data_two_class_each, create_client_data_ch_minst, create_client_data, \
    create_client_data_texas100
from scripts.mia_attacks import membership_inference_attack
from scripts.models import custom_model_ch_minst, create_model_softmax, create_purchase_classifier, vgg19_scratch
from scripts.utility import federated_averaging, draw_overlap_histogram, analysis_differences, plot_hist, \
    class_wise_analysis, get_gradient_norm, get_gradient_norm_data, plot_dist, plot_ge_cdf, plot_prediction_probability
from scripts.utility import visualize_decision_boundary, plot_generalization_error
from scripts.analysis import BaseAnalyzer
from scripts.federated_data import DataContainer
import matplotlib.pyplot as plt

# def plot_ge_cdf_combined(generalization_errors, file_name):
#     # Assuming `generalization_errors` is a dictionary where the keys are the class labels and the values are the generalization errors for each class
#     errors = np.array(list(generalization_errors))
#     # Calculate the sorted values and cumulative probabilities
#     sorted_errors = np.sort(errors)
#     cumulative_probs = np.arange(len(errors)) / float(len(errors) - 1)
#     # Plot the CDF
#     plt.plot(sorted_errors, cumulative_probs)
#     plt.xlabel('Generalization error')
#     plt.ylabel('Cumulative probability')
#     # plt.title('CDF of Generalization Error')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(file_name, dpi=150, format='png')
#     # plt.show()
#     plt.close()

def plot_ge_cdf_combined(generalization_errors1, generalization_errors2, file_name):
    # Assuming `generalization_errors` is a dictionary where the keys are the class labels and the values are the generalization errors for each class
    errors1 = np.array(list(generalization_errors1))
    errors2 = np.array(list(generalization_errors2))

    # Calculate the sorted values and cumulative probabilities for the first set of errors
    sorted_errors1 = np.sort(errors1)
    cumulative_probs1 = np.arange(len(errors1)) / float(len(errors1) - 1)

    # Calculate the sorted values and cumulative probabilities for the second set of errors
    sorted_errors2 = np.sort(errors2)
    cumulative_probs2 = np.arange(len(errors2)) / float(len(errors2) - 1)

    # Plot the CDF for the first set of errors
    plt.plot(sorted_errors1, cumulative_probs1, label='no defense')

    # Plot the CDF for the second set of errors
    plt.plot(sorted_errors2, cumulative_probs2, label='our defense')

    plt.xlabel('Generalization error')
    plt.ylabel('Cumulative probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=150, format='png')
    plt.close()

ch_mnist = ('ch-mnist-custom/member-shield/member-shield-kl-.70_ch-minst_custom_soft_0.7_el_sgd_0.001_0.99_grid-search_val_loss_Yes_threshold-entropy-knn-lr_entire-data.txt.xlsx',
            'ch-mnist-custom/no defense /no-defense_ch-minst_custom_hard_cce_sgd_0.001_0.99_val_loss_No_both-entropy-knn-lr_entire-data.txt.xlsx')

cifar10_custom = ('cifar10-custom/member-shield/member-shield-ce-.80_cifar10_custom_soft_0.8_el_sgd_0.001_0.99_grid-search_val_loss_Yes_threshold-entropy-knn-lr_entire-data.txt.xlsx',
                  'cifar10-custom/no defense/no-defense_cifar10_custom_hard_cce_sgd_0.001_0.99_val_loss_No_both-entropy-knn-lr_entire-data.txt.xlsx')
purchase100 =('purchase100-custom/member-shield/member-shield-ce-.80_texas100_custom_soft_0.8_el_sgd_0.001_0.99_grid-search_val_loss_Yes_threshold-entropy-knn-lr_entire-data.txt.xlsx',
              'purchase100-custom/no-defense/no-defense_texas100_custom_hard_cce_sgd_0.001_0.99_val_loss_No_both-entropy-knn-lr_entire-data.txt.xlsx')
cifar10_vgg19 = ('cifar10-vgg19/member-shield/member-shield-kl-.70_cifar10_vgg19_soft_0.7_el_sgd_0.001_0.99_grid-search_val_loss_Yes_threshold-entropy-knn-lr_entire-data.txt.xlsx',
                 'cifar10-vgg19/no defense/no-defense_cifar10_vgg19_hard_cce_sgd_0.001_0.99_val_loss_No_both-entropy-knn-lr_entire-data.txt.xlsx')
import pandas as pd
# Specify the path to your Excel file
file_path_se = cifar10_vgg19[0]
file_path_base = cifar10_vgg19[1]
# Read the Excel file
xls_se = pd.ExcelFile(file_path_se)
xls_base = pd.ExcelFile(file_path_base)

# Now you can read all the sheets in the file
for i in range(1, 6):
    #df1 =pd.read_excel(file_path_se, 'c{}'.format(i))
    df1 = pd.read_excel(file_path_base, 'c{}'.format(i))
    df1_acc = pd.read_excel(file_path_base, 'c{}_ge_acc'.format(i))
    df1_acc.drop('Unnamed: 0', axis=1, inplace=True)
    #print(df1_acc[1].values)
    df1_loss = pd.read_excel(file_path_base, 'c{}_ge_loss'.format(i))
    df2 = pd.read_excel(file_path_se, 'c{}'.format(i))
    df2_acc = pd.read_excel(file_path_se, 'c{}_ge_acc'.format(i))
    df2_acc.drop('Unnamed: 0', axis=1, inplace=True)
    #print(df2_acc[1].values)
    df2_loss = pd.read_excel(file_path_se, 'c{}_ge_loss'.format(i))
    for round in range(10):
        # print(df1_acc.iloc[round].values)
        # print(df2_acc.iloc[round].values)
        # plot_ge_cdf(df1_acc.iloc[round].values, 'se_acc.png')
        # plot_ge_cdf(df2_acc.iloc[round].values, 'base_acc.png')
        plot_ge_cdf_combined(df1_acc.iloc[round].values, df2_acc.iloc[round].values, '{}_{}_{}_acc.png'.format('cifar10-vgg19',round, i))


    # Print the data
    # print(df)
    # print(df_acc)
    # print(df_loss)

print('finished')


