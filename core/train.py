
import logging
import os
import tensorflow as tf
from .utils import get_loss_fn, get_metric, get_optimizer

logger = tf.get_logger()
logger.setLevel(logging.INFO)

class BaseTrainer:
    """Abstraction to train, track and save models

    """

    def __init__(self, config, model, dataset):
        """Initialise Trainer

        :param config: experiment config
        :param model: Keras sequential model instance
        :param dataset: Dataset instance
        :param tracker: Tracker instance
        """
        self.config = config
        self.model = model
        self.dataset = dataset

        self.exp_dir = self.config['Experiment']['id']

    def train(self):
        """train the model with logs and checkpoints

        :return: train history object
        """

        # callbacks = self.init_callbacks()
        #
        # # todo: make more generic
        # if self.pretrained:
        #     self.load_pretrained()

        # Compile the model

        self.model.compile(optimizer=get_optimizer(self.config['Trainer']['optimizer'], self.config['Trainer']['lr']),
                           loss=get_loss_fn(self.config['Trainer']['loss_fn']),
                           metrics=[get_metric(metric, self.config)
                                    for metric in self.config['Trainer']['eval_metrics']]
                           )

        logger.info("Model Compilation complete")

        history = self.model.fit(
            self.dataset.train_gen,
            verbose=1,
            epochs=self.config['Trainer']['epochs'],
            validation_data=self.dataset.val_gen,
        )

        self.trained = True
        logger.info("Model training complete")

    def save(self, save_format: str = 'tf_pb'):
        """Save the final trained model

        :return:
        """
        if self.trained:
            save_path = os.path.join(self.config['Trainer']['save_dir'], self.exp_dir, "saved_model")

            if save_format == 'tf_pb' or save_format == 'both':
                self.model.save(save_path)

            logger.info("Model saved to {}".format(save_path))
        else:
            logger.error('\nModel not trained!')
            raise ValueError