import argparse
# Initialize the pars

parser = argparse.ArgumentParser(description="A simple command line argument parser")

# Add the arguments
parser.add_argument("in_features_dim", help="The dimension of the feature vector/embedding in input", type=int, default=1024)
parser.add_argument("num_classes_target", help="Number of classes of the target", type=int, default=20)
parser.add_argument("num_classes_source", help="Number of classes of the source", type=int, default=20)
parser.add_argument("action", help="Train or validate", type=str, default="train")
parser.add_argument("resume_from", help="Checkpoint path if needed", default=None, type=str)
parser.add_argument("num_iter", help="Number of iterations for training", default=None, type=int)
parser.add_argument("total_batch", type=int, default=256)
parser.add_argument("batch_size", type=int, default=64)
parser.add_argument("lr_step", help="At which iteration to decrease learning rate", type=int, default=3000)
parser.add_argument("log_dir", help="Where to store file for log and results", type=str, default=".")
parser.add_argument("lr", help="Learning rate of the task", type=float, default=0.01)
parser.add_argument("weight_decay", help="Weight decay for regularisation", type=float, default=1e-6)
parser.add_argument("sgd_momentum", help="Momentum of the sgd optimiser", type=float, default=1e-4)
parser.add_argument("experiment_dir", help="Directory where to store model if needed", type=str, default=None)
parser.add_argument("remove_window_domain_classifier", help="Removes the window domain classifier", action='store_true', default=False)
parser.add_argument("remove_token_domain_classifier", help="Removes the token domain classifier", action='store_true', default=False)
parser.add_argument("dropout", help="Dropout of fully connected layers", type=float, default=0.5)
parser.add_argument("window_size", help="Length of the context window", type=str, default=5)
parser.add_argument("beta_window", help="GRL parameter for window", type=float, default=0.75)
parser.add_argument("beta_token", help="GRL parameter for token", type=float, default=0.75)
parser.add_argument("path_source_embeddings", default='./source/embeddings.pt', type=str)
parser.add_argument("path_source_labels", default='./source/labels.pt', type=str)
parser.add_argument("path_target_embeddings", default='./target/train/embeddings.pt', type=str)
parser.add_argument("path_target_labels", default='./target/train/labels.pt', type=str)
parser.add_argument("path_target_val_embeddings", default='./target/val/embeddings.pt', type=str)
parser.add_argument("path_target_val_labels", default='./target/val/labels.pt', type=str)

# Parse the arguments
args = parser.parse_args()