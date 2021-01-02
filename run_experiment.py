import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument("--config", type=str, dest='config', required=True, help="Experiment Configuration")

    parser.add_argument("--summary", default=False, help="Only show summary", dest='summary', action='store_true')

    parser.add_argument("--save_df", default=False, help="Save Dataframe", dest='save_df', action='store_true')
    parser.add_argument("--df_name", type=str, dest='df_name',
                        required=('--save_df' in sys.argv or '--save_crop_df' in sys.argv or '--save_tf_dataset' in sys.argv),
                        help="Dataframe filename")
    parser.add_argument("--df_fraction", type=float, dest='df_fraction', required='--save_df' in sys.argv,
                        help="Dataframe size")

    parser.add_argument("--train", default=False, help="Train model", dest='train', action='store_true')

    parser.add_argument("--evaluate", default=False, help="Evaluate model", dest='evaluate', action='store_true')
    parser.add_argument("--export_frozen_graph", default=False,
                        help="Export Frozen Graph", dest='export_frozen_graph', action='store_true')
    parser.add_argument("--save_crop_df", default=False, help="Save Dataframe", dest='save_crop_df', action='store_true')
    parser.add_argument("--save_tf_dataset", default=False, help="Save Dataframe", dest='save_tf_dataset', action='store_true')
    parser.add_argument("--saved_model", type=str, dest='saved_model',
                        required=('--convert' in sys.argv or '--evaluate' in sys.argv),
                        help="Saved model name")

    return parser.parse_args()


if __name__ == '__main__':
    from core.dataset import BaseDataset
    from core.utils import read_config

    args = parse_args()

    # Load config
    exp_config = read_config(args.config)

    dataset = BaseDataset(config=exp_config)

    dataset.prepare_data()

    vocabulary = set()
    for key in dataset.descriptions.keys():
        [vocabulary.update(d.split()) for d in dataset.descriptions[key]]
    print('Original Vocabulary Size: %d' % len(vocabulary))
    # 8763

    print('Dataset: %d' % len(dataset.train_text))
    print('Dataset: %d' % len(dataset.val_text))
    print('Dataset: %d' % len(dataset.test_text))

    print('Extracted Features: %d' % len(dataset.features))