import dp_accounting
# from scripts.mia_attacks import perform_mia
import math
import matplotlib.pyplot as plt
import numpy as np
from absl import app
import tensorflow as tf
from scipy import stats
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from scipy.stats import entropy
from tensorflow.keras.callbacks import EarlyStopping


def calculate_noise(n, batch_size, target_epsilon, epochs, delta, noise_lbd):
    """Compute noise based on the given hyperparameters."""
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise app.UsageError('n must be larger than the batch size.')
    orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
              list(range(5, 64)) + [128, 256, 512])
    steps = int(math.ceil(epochs * n / batch_size))

    def make_event_from_noise(sigma):
        t = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps)
        return t

    def make_accountant():
        return dp_accounting.rdp.RdpAccountant(orders)

    accountant = make_accountant()
    accountant.compose(make_event_from_noise(noise_lbd))
    init_epsilon = accountant.get_epsilon(delta)

    if init_epsilon < target_epsilon:  # noise_lbd was an overestimate
        # print('noise_lbd too large for target epsilon.')
        return 0

    target_noise = dp_accounting.calibrate_dp_mechanism(
        make_accountant, make_event_from_noise, target_epsilon, delta,
        None)

    # print(
    #     'DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
    #     ' over {} steps satisfies'.format(100 * q, target_noise, steps),
    #     end=' ')
    # print('differential privacy with eps = {:.3g} and delta = {}.'.format(
    #     target_epsilon, delta))
    # file.write(
    #     'DP-SGD with sampling rate = {:.3g} and noise_multiplier = {} iterated over {} steps satisfies differential privacy with eps = {:.3g} and delta = {}.\n\n'.format(
    #         100 * q, target_noise, steps, target_epsilon, delta))
    return target_noise


def entropy(preds, axis=0):
    logp = np.log(preds)
    entropy = np.sum(-preds * logp, axis=axis)
    return entropy


# def modified_entropy(prob_logit, y_batch_train, axis=0):
#     true_class_index = tf.argmax(y_batch_train, axis=-1)
#     # Get the softmax probability of the true class
#     true_class_prob = tf.gather(prob_logit, true_class_index, batch_dims=1)
#     # Split the tensor into two parts: before and after the given index
#     prob_before_index = prob_logit[:true_class_index]
#     prob_after_index = prob_logit[true_class_index + 1:]
#     # Concatenate the two parts to get the probabilities of all other classes
#     prob_except_index = tf.concat([prob_before_index, prob_after_index], axis=0)
#     # logp = np.log(preds)
#     # entropy = np.sum(-preds * logp, axis=axis)
#     m_entropy = -(1 - true_class_prob) * tf.math.log(true_class_prob) - tf.reduce_sum(
#         prob_except_index * tf.math.log(1 - prob_except_index), axis=0)
#     return entropy


def modified_entropy(prob_logit, y_batch_train):
    true_class_index = tf.argmax(y_batch_train, axis=-1)
    # Get the softmax probability of the true class
    true_class_prob = tf.gather(prob_logit, true_class_index, batch_dims=1)

    # Create a mask that excludes the true class index
    mask = tf.one_hot(true_class_index, depth=prob_logit.shape[-1], dtype=tf.bool, on_value=False, off_value=True)
    # Apply the mask to get the probabilities of all other classes
    prob_except_index = tf.boolean_mask(prob_logit, mask)

    # Reshape the tensor to match the original shape, but with one less class
    new_shape = tf.concat([tf.shape(prob_logit)[:-1], [prob_logit.shape[-1] - 1]], axis=0)
    prob_except_index = tf.reshape(prob_except_index, new_shape)

    # Compute the modified entropy
    m_entropy = -(1 - true_class_prob) * tf.math.log(true_class_prob) - tf.reduce_sum(
        prob_except_index * tf.math.log(1 - prob_except_index), axis=-1)

    return m_entropy


def get_prob_except_index(prob_logit, exclude_index):
    # Split the tensor into two parts: before and after the given index
    prob_before_index = prob_logit[:exclude_index]
    prob_after_index = prob_logit[exclude_index + 1:]

    # Concatenate the two parts to get the probabilities of all other classes
    prob_except_index = tf.concat([prob_before_index, prob_after_index], axis=0)

    return prob_except_index


def get_top1(num_classes, entropy_threshold, reduced_prob=0.01):
    # reduced_prob : for reducing top-1 class's probability
    true_target = 1
    preds = np.zeros(num_classes)
    preds[true_target] = 1.
    while (True):
        preds[true_target] -= reduced_prob
        preds[:true_target] += reduced_prob / (num_classes - 1)
        preds[true_target + 1:] += reduced_prob / (num_classes - 1)
        if (entropy(preds) >= entropy_threshold):
            break
    return preds[true_target], preds[true_target + 1]


def get_soft_labels(train_label, num_classes, top1, uniform_non_top1):
    # new_soft_label = np.zeros( (train_label.shape[0], num_classes) )
    copy_train_label = train_label
    for i in range(len(train_label)):
        label_index = np.argmax(train_label[i])
        copy_train_label[i][label_index] = top1
        copy_train_label[i][:label_index] = uniform_non_top1
        copy_train_label[i][label_index + 1:] = uniform_non_top1
    return copy_train_label


def magnitude_shift(model):
    # Define the range for random values
    lower_bound = 0.75
    upper_bound = 0.85

    # Generate random values within the specified range for the averaged weights
    scaling_factor = [
        tf.random.uniform(var.shape, minval=lower_bound, maxval=upper_bound) for var in model.trainable_weights
    ]
    model_weights = [weight for weight in model.get_weights()]
    scaled_weightes = []
    for i, weight in enumerate(model_weights):
        scaled_weightes.append(weight * scaling_factor[i].numpy())
    model.set_weights(scaled_weightes)
    return model


def federated_averaging(global_model, client_models, num_clients):
    averaged_weights = [tf.zeros_like(var) for var in global_model.trainable_weights]
    for client_model in client_models:
        for i in range(len(averaged_weights)):
            averaged_weights[i] += client_model.trainable_weights[i] / num_clients
    global_model.set_weights(averaged_weights)
    return global_model


def federated_averaging_resnet(global_model, client_models, num_clients):
    actual_weights = [weight for weight in global_model.weights]
    averaged_weights = [tf.zeros_like(var) for var in global_model.trainable_weights]
    for client_model in client_models:
        for i in range(len(averaged_weights)):
            averaged_weights[i] += client_model.trainable_weights[i] / num_clients
    count = 0
    for idx, weight in enumerate(actual_weights):
        if weight.trainable:
            actual_weights[i] = averaged_weights[count]
            count = count + 1
    global_model.set_weights(actual_weights)
    return global_model


def federated_average(global_model, models):
    averaged_weights = [tf.zeros_like(var) for var in global_model.weights]
    # averaged_weights = [tf.zeros_like(var) if var.trainable else var.read_value() for var in models.weights]
    averaged_weights = [weight for weight in models[0].get_weights()]
    # averaged_weights = [w for w in global_model.get_weights()]

    for i, weight in enumerate(models[0].weights):
        # if weight.trainable:
        for j in range(1, len(models)):
            averaged_weights[i] += models[j].get_weights()[i] / len(models)
    global_model.set_weights(averaged_weights)
    return global_model


def create_deep_copy(model):
    model_config = model.get_config()
    new_model = tf.keras.models.Model.from_config(model_config)
    new_model.set_weights(model.get_weights())
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    new_model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return new_model


def vectorize_parameters(model):
    flattened_params = []
    for layer in model.layers:
        # Check if the layer has trainable weights (Conv2D, Dense, etc.)
        if layer.trainable_weights:
            # Flatten the weights and biases of the layer
            params = [param.numpy().flatten() for param in layer.trainable_weights]
            # Concatenate the flattened parameters into a single vector
            flattened_params.extend(params)

    # Convert the list of flattened parameters to a single 1D NumPy array
    flattened_params = np.concatenate(flattened_params)
    # The variable flattened_params now contains all the model's parameters as a 1D array
    print("Total number of parameters:", len(flattened_params))
    return flattened_params


def reshape_parameters(model, param_vector):
    weights = model.get_weights()
    start_index = 0
    end_index = None
    train_weight = []

    for layer in model.layers:
        # Check if the layer has trainable weights (Conv2D, Dense, etc.)
        if layer.trainable_weights:
            for param in layer.trainable_weights:
                end_index = tf.reduce_prod(param.shape)
                reshape_param = np.array(param_vector[start_index:start_index + end_index]).reshape(param.shape)
                train_weight.append(tf.convert_to_tensor(reshape_param, dtype=tf.float32))
                # train_weight.append(reshape_param)
                start_index = end_index

    return train_weight


def de_generated_weight(model, weight_list, train_data, test_data):
    # Define the objective function to be optimized (example: sphere function)
    X_test, y_test = test_data[0], test_data[1]
    X_train, y_train = train_data[0], train_data[1]
    # DE hyperparameters
    population = weight_list
    population_size = len(weight_list)
    max_generations = 10
    F = 0.8  # Scaling factor
    CR = 0.5  # Crossover probability

    def objective_function(parameters):
        reshaped_weight = reshape_parameters(model, parameters)
        val_eva = model.evaluate(X_test, y_test, verbose=0)
        m_auc, m_adv = perform_mia(1, model, X_train, X_test, y_train, y_test)
        return m_adv

    # define crossover operation
    def crossover(mutated, target, dims, cr):
        # generate a uniform random value for every dimension
        p = np.random.rand(dims)
        # generate trial vector by binomial crossover
        trial = np.array([mutated[i] if p[i] < cr else target[i] for i in range(dims)])
        return trial

    def differential_evolution(pop_size, iter, F, cr):
        # initialise population of candidate solutions randomly within the specified bounds
        # pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
        # evaluate initial population of candidate solutions
        obj_all = [objective_function(ind) for ind in population]
        # find the best performing vector of initial population
        best_vector = population[np.argmin(obj_all)]
        best_obj = min(obj_all)
        prev_obj = best_obj
        # run iterations of the algorithm
        for i in range(iter):
            # iterate over all candidate solutions
            for j in range(pop_size):
                # choose three candidates, a, b and c, that are not the current one
                candidates = [candidate for candidate in range(pop_size) if candidate != j]
                # a, b, c = population[np.random.choice(candidates, 3, replace=False)]
                a, b, c = [population[i] for i in np.random.choice(candidates, 3, replace=False)]
                # perform mutation
                mutated = a + F * (b - c)
                # check that lower and upper bounds are retained after mutation
                # mutated = check_bounds(mutated, bounds)
                # perform crossover
                trial = crossover(mutated, population[j], len(mutated), cr)

                # compute objective function value for target vector
                obj_target = objective_function(population[j])
                # compute objective function value for trial vector
                obj_trial = objective_function(trial)
                # perform selection
                if obj_trial < obj_target:
                    # replace the target vector with the trial vector
                    population[j] = trial
                    # store the new objective function value
                    obj_all[j] = obj_trial
            # find the best performing vector at each iteration
            best_obj = min(obj_all)
            # store the lowest objective function value
            if best_obj < prev_obj:
                best_vector = population[np.argmin(obj_all)]
                prev_obj = best_obj
            # report progress at each iteration
            print('Iteration: %d f([%s]) = %.5f' % (i, np.around(best_vector, decimals=5), best_obj))
        return [best_vector, best_obj]

    return differential_evolution(population_size, max_generations, F, CR)


def de_generated_weight_gpu(model, weight_list, train_data, test_data):
    # Define the objective function to be optimized (example: sphere function)
    X_test, y_test = test_data[0], test_data[1]
    X_train, y_train = train_data[0], train_data[1]

    # DE hyperparameters
    population = weight_list
    population_size = len(weight_list)
    max_generations = 10
    F = 0.8  # Scaling factor
    CR = 0.5  # Crossover probability

    # Convert data to TensorFlow tensors if not already
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

    # Define the objective function using TensorFlow
    @tf.function
    def objective_function(parameters):
        with tf.device('/GPU:0'):
            reshaped_weight = reshape_parameters(model, parameters)
            val_eva = model.evaluate(X_test, y_test, verbose=0)
            m_auc, m_adv = perform_mia(1, model, X_train, X_test, y_train, y_test)
            return m_adv

    # Define crossover operation
    def crossover(mutated, target, dims, cr):
        p = tf.random.uniform(shape=(dims,), dtype=tf.float32)
        trial = tf.where(p < cr, mutated, target)
        return trial

    def differential_evolution(pop_size, iter, F, cr):
        with tf.device('/GPU:0'):
            obj_all = [objective_function(ind) for ind in population]
            best_vector = population[np.argmin(obj_all)]
            best_obj = min(obj_all)
            prev_obj = best_obj

            for i in range(iter):
                for j in range(pop_size):
                    candidates = [candidate for candidate in range(pop_size) if candidate != j]
                    a, b, c = [population[i] for i in np.random.choice(candidates, 3, replace=False)]
                    mutated = a + F * (b - c)
                    trial = crossover(mutated, population[j], len(mutated), cr)
                    obj_target = objective_function(population[j])
                    obj_trial = objective_function(trial)

                    if obj_trial < obj_target:
                        population[j] = trial
                        obj_all[j] = obj_trial

                best_obj = min(obj_all)

                if best_obj < prev_obj:
                    best_vector = population[np.argmin(obj_all)]
                    prev_obj = best_obj
                print('Iteration: %d f([%s]) = %.5f' % (i, np.around(best_vector, decimals=5), best_obj))

            return [best_vector, best_obj]

    return differential_evolution(population_size, max_generations, F, CR)


def draw_histogram(training_losses, test_losses):
    # Your list containing categorical cross-entropy loss for the training and test datasets
    # training_losses = [0.5, 0.8, 1.2, 0.9, 1.5, 0.7, 1.0]  # Replace with your actual training losses
    # test_losses = [0.6, 0.7, 1.0, 1.1, 1.3, 0.9]  # Replace with your actual test losses

    # Plotting histograms for training and test losses
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(training_losses, bins=10, alpha=0.7, color='blue', density=True)
    plt.title('Training Set Loss')
    plt.xlabel('Loss')
    plt.ylabel('Normalized Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(test_losses, bins=10, alpha=0.7, color='green', density=True)
    plt.title('Test Set Loss')
    plt.xlabel('Loss')
    plt.ylabel('Normalized Frequency')

    plt.tight_layout()
    plt.show()


def draw_overlap_histogram(method, round, client, training_losses, test_losses, tag='loss'):
    plt.figure(figsize=(8, 6))
    plt.hist(training_losses, bins=10, alpha=0.5, color='blue', density=True, label='member')
    plt.hist(test_losses, bins=10, alpha=0.5, color='green', density=True, label='non-member')
    plt.title('member and non-member {}'.format(tag))
    plt.xlabel(tag)
    plt.ylabel('normalized frequency')
    plt.legend()
    # Save the plot as an image (e.g., PNG)
    plt.savefig('{}_{}_r{}_c{}.png'.format(method, tag, round, client), bbox_inches='tight')
    # plt.show(block=False)
    plt.close()


def plot_hist(values, names, method, round, client, tag='loss'):
    plt.figure()
    bins = np.histogram(np.hstack(values), bins=50)[1]
    for val, name in zip(values, names):
        plt.hist(val, bins=bins, alpha=0.5, label=name)
    # plt.title('member and non-member {}'.format(tag))
    plt.xlabel(tag)
    plt.ylabel('frequency')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig(save_file, dpi=150, format='png')
    plt.savefig('{}_{}_r{}_c{}-new.png'.format(method, tag, round, client), dpi=150, format='png')
    plt.close()


def plot_dist(member_dist, non_member_dist, method, round, client, tag='Prediction probability'):
    # Plot the distribution of entropies
    # sns.distplot(entropies_train, hist=False, kde=True, kde_kws={'linewidth': 2}, label='member')
    # sns.distplot(entropies_test, hist=False, kde=True, kde_kws={'linewidth': 2}, label='non-member')
    # Plot the distribution of these probabilities
    sns.distplot(member_dist, hist=False, kde=True, kde_kws={'linewidth': 2}, label='member')
    sns.distplot(non_member_dist, hist=False, kde=True, kde_kws={'linewidth': 2}, label='non-member')
    # plt.hist(true_class_probabilities, bins=30, density=True)
    # plt.hist(true_class_probabilities, bins=50, alpha=0.5, color='blue', density=True, label='member')
    # plt.hist(true_class_probabilities_test, bins=50, alpha=0.5, color='green',density=True,  label='non-member')
    # plt.title('Probability Distribution of True Class')
    plt.xlabel(tag)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('{}_{}_r{}_c{}.png'.format(method, tag, round, client), dpi=150, format='png')
    # plt.show()
    plt.close()


def plot_ge_cdf(generalization_errors, file_name):
    # Assuming `generalization_errors` is a dictionary where the keys are the class labels and the values are the generalization errors for each class
    errors = np.array(list(generalization_errors))
    # Calculate the sorted values and cumulative probabilities
    sorted_errors = np.sort(errors)
    cumulative_probs = np.arange(len(errors)) / float(len(errors) - 1)
    # Plot the CDF
    plt.plot(sorted_errors, cumulative_probs)
    plt.xlabel('Generalization error')
    plt.ylabel('Cumulative probability')
    # plt.title('CDF of Generalization Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=150, format='png')
    # plt.show()
    plt.close()


def laplace_smooth(probabilities, alpha=1.0):
    total_count = len(probabilities)
    smoothed_probabilities = (probabilities + alpha) / (total_count + alpha * total_count)
    return smoothed_probabilities


def analysis_differences(model, X_train, X_test, y_train, y_test, batch_size):
    logit_train = model.predict(X_train, batch_size=batch_size)
    logit_test = model.predict(X_test, batch_size=batch_size)
    logit_train = tf.nn.softmax(logit_train, axis=-1)
    logit_test = tf.nn.softmax(logit_test)
    logit_train_smooth = np.array([laplace_smooth(probabilities) for probabilities in logit_train])
    logit_test_smooth = np.array([laplace_smooth(probabilities) for probabilities in logit_test])
    entropy_train = -tf.reduce_sum(logit_train_smooth * tf.math.log(logit_train_smooth), axis=-1)
    entropy_test = -tf.reduce_sum(logit_test_smooth * tf.math.log(logit_test_smooth), axis=-1)

    # m_entropy_train = modified_entropy(logit_train, y_train)
    # m_entropy_test = modified_entropy(logit_test, y_test)

    m_entropy_train = modified_entropy(logit_train_smooth, y_train)
    m_entropy_test = modified_entropy(logit_test_smooth, y_test)

    # entropy_train = -tf.reduce_sum(logits_train * tf.math.log(logits_train), axis=-1)
    # entropy_test = -tf.reduce_sum(logits_test * tf.math.log(logits_test), axis=-1)
    cce = tf.keras.backend.categorical_crossentropy
    constant = tf.keras.backend.constant
    loss_train = cce(constant(y_train), constant(logit_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test), constant(logit_test), from_logits=False).numpy()
    return loss_train, loss_test, entropy_train, entropy_test, m_entropy_train, m_entropy_test


def get_class_wise_samples(X, y):
    # Convert one-hot encoded labels to class indices
    class_indices = tf.argmax(y, axis=1)

    # Get unique class indices
    unique_classes = tf.unique(class_indices).y

    # Initialize a dictionary to hold the samples for each class
    class_samples = {}
    class_labels = {}

    # Loop over each unique class
    for class_index in unique_classes:
        # Use boolean indexing to filter the samples that belong to the current class
        class_samples[class_index.numpy()] = X[class_indices == class_index]
        class_labels[class_index.numpy()] = y[class_indices == class_index]

    return [class_samples, class_labels]


# Usage:
# class_samples = retrieve_samples_classwise(X, y)
# Now, class_samples[i] will give you all samples that belong to class i

def get_gradient_norm(model, loss_fn, x_sample, y_sample):
    # Convert the sample to a batch with a single item
    x_batch = tf.expand_dims(x_sample, 0)
    y_batch = tf.expand_dims(y_sample, 0)

    # Compute the loss for the sample
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        y_pred = model(x_batch, training=True)
        loss = loss_fn(y_batch, y_pred)
    # Compute the gradients of the loss with respect to the model's parameters
    gradients = tape.gradient(loss, model.trainable_variables)

    # Compute the norm of the gradient
    gradient_norm = tf.norm([tf.norm(grad) for grad in gradients])

    # print('Gradient norm:', gradient_norm)
    return gradient_norm


def get_gradient_norm_data(model, loss_fn, data, labels):
    with tf.GradientTape() as tape:
        predictions = model(data)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_norms = [tf.norm(gradient).numpy() for gradient in gradients]
    # average_norm = sum(gradient_norms) / len(gradient_norms)
    return gradient_norms


def class_wise_analysis(model, X_train, X_test, y_train, y_test, batch_size, class_label):
    generalization_errors = {}
    generalization_errors_loss = {}

    class_wise_train = get_class_wise_samples(X_train, y_train)
    class_wise_test = get_class_wise_samples(X_test, y_test)

    logit_train_ten = model.predict(class_wise_train[0][class_label], batch_size=batch_size)
    logit_test_ten = model.predict(class_wise_test[0][class_label], batch_size=batch_size)
    single_class_prob_train, single_class_prob_test = tf.nn.softmax(logit_train_ten, axis=-1), tf.nn.softmax(
        logit_test_ten)

    # Make predictions
    for key, X_train_class in class_wise_train[0].items():
        # train_preds = model.predict(X_train_class)
        # test_preds = model.predict(class_wise_test[0][key])
        # # Calculate accuracies
        # train_acc = accuracy_score(class_wise_train[1][key], train_preds)
        # test_acc = accuracy_score(class_wise_test[1][key], test_preds)

        train_res = model.evaluate(X_train_class, class_wise_train[1][key], verbose=0)
        if key in class_wise_test[0]:
            test_res = model.evaluate(class_wise_test[0][key], class_wise_test[1][key], verbose=0)
        else:
            test_res = model.evaluate(class_wise_test[0][key - 1], class_wise_test[1][key - 1], verbose=0)
        # Calculate and store the generalization error for the current class
        generalization_errors[key] = np.abs(test_res[1] - train_res[1])
        generalization_errors_loss[key] = np.abs(test_res[0] - train_res[0])

    # return generalization_errors.values(), generalization_errors_loss.values()

    # plot_ge_cdf(generalization_errors.values())
    # plot_ge_cdf(generalization_errors_loss.values())

    logit_train = model.predict(X_train, batch_size=batch_size)
    logit_test = model.predict(X_test, batch_size=batch_size)
    logit_train = tf.nn.softmax(logit_train, axis=-1)
    logit_test = tf.nn.softmax(logit_test)

    true_class_indices = tf.argmax(y_train, axis=-1)
    # Get the probabilities of the true class
    true_class_probabilities = tf.gather(logit_train, true_class_indices, batch_dims=1).numpy()

    true_class_indices_test = tf.argmax(y_test, axis=-1)
    # Get the probabilities of the true class
    true_class_probabilities_test = tf.gather(logit_test, true_class_indices_test, batch_dims=1).numpy()

    # Calculate entropy for each prediction
    # entropies_train = [entropy(prediction) for prediction in logit_train]
    # entropies_test = [entropy(prediction) for prediction in logit_test]

    # Plot the distribution of entropies
    # sns.distplot(entropies_train, hist=False, kde=True, kde_kws={'linewidth': 2}, label='member')
    # sns.distplot(entropies_test, hist=False, kde=True, kde_kws={'linewidth': 2}, label='non-member')
    # Plot the distribution of these probabilities
    # plt.hist(true_class_probabilities, bins=30, density=True)
    # plt.hist(true_class_probabilities, bins=50, alpha=0.5, color='blue', density=True, label='member')
    # plt.hist(true_class_probabilities_test, bins=50, alpha=0.5, color='green',density=True,  label='non-member')

    # sns.distplot(true_class_probabilities, hist=False, kde=True, kde_kws={'linewidth': 2}, label='member')
    # sns.distplot(true_class_probabilities_test, hist=False, kde=True, kde_kws={'linewidth': 2}, label='non-member')
    # plt.title('Probability Distribution of True Class')
    # plt.xlabel('Probability')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.savefig('probability distribution-member-shield-texas-.png', bbox_inches='tight')
    # plt.show()
    # plt.close()

    # logit_train_smooth = np.array([laplace_smooth(probabilities) for probabilities in logit_train])
    # logit_test_smooth = np.array([laplace_smooth(probabilities) for probabilities in logit_test])
    # entropy_train = -tf.reduce_sum(logit_train_smooth * tf.math.log(logit_train_smooth), axis=-1)
    # entropy_test = -tf.reduce_sum(logit_test_smooth * tf.math.log(logit_test_smooth), axis=-1)

    # true_class_index_train = tf.argmax(y_train, axis=-1)
    # # Get the softmax probability of the true class
    # true_class_prob_train = tf.gather(logit_train, true_class_index, batch_dims=1)

    # entropy_train = -tf.reduce_sum(logits_train * tf.math.log(logits_train), axis=-1)
    # entropy_test = -tf.reduce_sum(logits_test * tf.math.log(logits_test), axis=-1)
    # cce = tf.keras.backend.categorical_crossentropy
    # constant = tf.keras.backend.constant
    # loss_train = cce(constant(y_train), constant(logit_train), from_logits=False).numpy()
    # loss_test = cce(constant(y_test), constant(logit_test), from_logits=False).numpy()
    return generalization_errors, generalization_errors_loss, true_class_probabilities, true_class_probabilities_test, single_class_prob_train, single_class_prob_test


def visualize_decision_boundary(softmax_output, data_split, method, training_round=None, client=None):
    # Assuming you have the softmax outputs for the test dataset stored in test_softmax
    # test_softmax should be a numpy array with shape (num_samples, num_classes)

    # Using t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_output = tsne.fit_transform(softmax_output)

    # Plot the t-SNE visualization
    plt.figure(figsize=(8, 6))
    c = np.argmax(softmax_output, axis=1)

    # Scatter plot the t-SNE transformed data
    plt.scatter(tsne_output[:, 0], tsne_output[:, 1], c=np.argmax(softmax_output, axis=1), cmap='viridis')
    plt.colorbar()
    # plt.title('{} : class discrimination for {} samples'.format(method, data_split))
    plt.savefig('{}_{}_r{}_c{}.png'.format(method, data_split, training_round, client), bbox_inches='tight')
    # plt.show()
    plt.close()


def perform_t_test(training_set_loss, test_set_loss):
    # Perform two-sample t-test
    t_statistic, p_value = stats.ttest_ind(training_set_loss, test_set_loss)

    # Print results
    print("T-Statistic:", t_statistic)
    print("P-Value:", p_value)

    # Check if the difference is statistically significant at a significance level of 0.05
    alpha = 0.05
    if p_value < alpha:
        print("The difference is statistically significant")
    else:
        print("The difference is not statistically significant")


def plot_cdf(test_accuracy, train_accuracy):
    # Step 1: Generate generalization error values (replace this with your own data)
    generalization_errors = np.random.normal(loc=0, scale=1, size=10)
    # Step 2: Calculate the CDF using NumPy
    sorted_errors = np.sort(generalization_errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    # Step 3: Visualize the CDF using Matplotlib
    plt.plot(sorted_errors, cumulative_prob, label='CDF')
    plt.title('Cumulative Distribution Function (CDF) of Generalization Error')
    plt.xlabel('Generalization Error')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_generalization_error(test_accuracy, train_accuracy, test_loss, train_loss, file_name):
    # Assuming 'history' is the returned object from the 'fit()' function
    # For example: history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(test_accuracy)
    plt.plot(train_accuracy)
    # plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Communication round')
    plt.legend(['member', 'non-member'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(test_loss)
    plt.plot(train_loss)
    # plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Communication round')
    plt.legend(['member', 'non-member'], loc='upper left')
    plt.tight_layout()
    plt.savefig(file_name, dpi=150, format='png')
    # plt.show()
    plt.close()


def plot_prediction_probability(probabilities, name):
    # Convert to numpy array for easier manipulation
    probabilities = np.array(probabilities)

    # Plotting
    plt.figure(figsize=(10, 5))

    # Loop over each sample and plot
    for i, sample_probabilities in enumerate(probabilities):
        plt.plot(range(len(sample_probabilities)), sample_probabilities)

    # plt.title('Class Probability Distribution of Test Samples')
    plt.xlabel('Class Label')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig(name + '.png', dpi=150, format='png')
    # plt.show()
    plt.close()


class custom_cce(tf.keras.losses.Loss):
    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        predicted_entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        # loss_fn = tf.keras.losses.KLDivergence()
        loss_value = loss_fn(y_true, y_pred)
        loss = loss_value - self.penalty * tf.reduce_mean(predicted_entropy)
        return loss
class CustomEarlyStopping(EarlyStopping):
    def __init__(self, initial_val_loss, **kwargs):
        super(CustomEarlyStopping, self).__init__(**kwargs)
        self.initial_val_loss = initial_val_loss

    def on_train_begin(self, logs=None):
        # Set the initial best to the initial validation loss
        self.best = self.initial_val_loss
        super(CustomEarlyStopping, self).on_train_begin(logs)

