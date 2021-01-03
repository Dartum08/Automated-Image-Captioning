import yaml
import tensorflow as tf

def read_config(path: str):
    with open(path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def get_loss_fn(loss_fn: str):
    if loss_fn == 'BinaryCrossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_fn == 'CategoricalCrossentropy':
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss_fn == 'SparseCategoricalCrossentropy':
        return 'sparse_categorical_crossentropy'
    elif loss_fn == 'mse':
        return loss_fn
    elif loss_fn == 'mae':
        return loss_fn
    else:
        raise ValueError


def get_optimizer(optimizer: str, learning_rate: float):
    if optimizer == 'sgd':
        return tf.keras.optimizers.SGD(lr=learning_rate)
    elif optimizer == 'adam':
        return tf.keras.optimizers.Adam(lr=learning_rate)
    elif optimizer == 'adagrad':
        return tf.keras.optimizers.Adagrad(lr=learning_rate)
    elif optimizer == 'rmsprop':
        return tf.keras.optimizers.RMSprop(lr=learning_rate)
    else:
        raise ValueError


def get_metric(metric: str, config):
    if metric == 'precision':
        return tf.metrics.Precision(thresholds=config['Trainer']['threshold'])
    elif metric == 'accuracy':
        return metric
    elif metric == 'recall':
        return tf.metrics.Recall(thresholds=config['Trainer']['threshold'])
    elif metric == 'mse':
        return tf.keras.metrics.MeanSquaredError()
    elif metric == 'mae':
        return tf.keras.metrics.MeanAbsoluteError()
    elif metric == 'rmse':
        return tf.keras.metrics.RootMeanSquaredError()
    else:
        print('No such metric available')
        raise ValueError
