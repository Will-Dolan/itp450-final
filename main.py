from pipeline import Pipeline
import argparse


def config_parser():
    configuration_parser = argparse.ArgumentParser()
    # defaults
    configuration_parser.add_argument("-e", "--epochs", type=int, default=30, help="Training Epochs")
    configuration_parser.add_argument("-smp", "--saved_model_pathway", type=str, default="", help="Saved Model Pathway")
    configuration_parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch Size")
    configuration_parser.add_argument("-ss", "--seq_size", type=int, default=128, help="Sequence Size")
    configuration_parser.add_argument("-nh", "--num_heads", type=int, default=12, help="Number of Attention Heads")
    configuration_parser.add_argument("-nl", "--num_layers", type=int, default=12, help="Number of Layers")

    # necessary configuration
    configuration_parser.add_argument("-s", "--seed", type=int, help="Randomizer Seed")

    return configuration_parser


if __name__ == '__main__':
    # pseudocode
    parser = config_parser()
    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.configure_gpus()
    pipeline.load_dataset()
    pipeline.load_model()
    pipeline.train_model()
    pipeline.test_model()