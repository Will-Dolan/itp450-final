from pipeline import Pipeline
import argparse

def config_parser():
	configuration_parser = argparse.ArgumentParser()
	# defaults
	configuration_parser.add_argument("-e", "--epochs", type=int, default=10, help="Training Epochs")
	configuration_parser.add_argument("-smp", "--saved_model_pathway", type=str, default="", help="Saved Model Pathway")
	configuration_parser.add_argument("-bs", "--batch_size", type=int, default=32,  help="Batch Size")

	# necessary configuration
	configuration_parser.add_argument("-s", "--seed", type=int, help="Randomizer Seed")
	configuration_parser.add_argument("-en", "--experiment_name", type=str, help="Experiment Name, for storage")
	configuration_parser.add_argument("-ilr", "--init_learning_rate", type=float,
									  help="Initial Learning Rate for the optimizer")

	return configuration_parser

if __name__=='__main__':
	# pseudocode
	parser = config_parser()
	args = parser.parse_args()


	pipeline = Pipeline()
	pipeline.configure_gpus()
	pipeline.load_dataset()
	pipeline.load_model()
	pipeline.train_model()
	pipeline.test_model()