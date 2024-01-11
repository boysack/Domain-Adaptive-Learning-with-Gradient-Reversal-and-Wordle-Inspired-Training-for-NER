import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from domain_adaptation_ner import DomainAdaptationNER
import argparse
import utils
from utils.logger import logger
from embeddingsDataLoader import EmbeddingDataset

# Initialize the parser
parser = argparse.ArgumentParser(description="A simple command line argument parser")

# Add the arguments
parser.add_argument("in_features_dim", help="The dimension of the feature vector/embedding in input", type=int)
parser.add_argument("num_classes_target", help="Number of classes of the target", type=int)
parser.add_argument("num_classes_source", help="Number of classes of the source", type=int)
parser.add_argument("action", help="Train or validate", type=str)
parser.add_argument("resume_from", help="Checkpoint path if needed", default=None, type=str)
parser.add_argument("num_iter", help="Number of iterations for training", default=None, type=int)
parser.add_argument("total_batch", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("lr_step", help="At which iteration to decrease learning rate", type=int)
parser.add_argument("log_dir", help="Where to store file for log and results", type=str)
parser.add_argument("lr", help="Learning rate of the task", type=float)
parser.add_argument("weight_decay", help="Weight decay for regularisation", type=float)
parser.add_argument("weight_decay", help="Weight decay for regularisation", type=float)
parser.add_argument("sgd_momentum", help="Momentum of the sgd optimiser", type=float)
parser.add_argument("experiment_dir", help="Directory where to store model if needed", type=str)
parser.add_argument("remove_window_domain_classifier", help="Removes the window domain classifier", action='Storetrue', default=False)
parser.add_argument("remove_token_domain_classifier", help="Removes the token domain classifier", action='Storetrue', default=False)
parser.add_argument("dropout", help="Dropout of fully connected layers", type=float)
parser.add_argument("window_size", help="Length of the context window", type=str)
parser.add_argument("beta_window", help="GRL parameter for window", type=float)
parser.add_argument("beta_token", help="GRL parameter for token", type=float)
parser.add_argument("path_source_embeddings", default='./source/embeddings.pt', type=str)
parser.add_argument("path_source_labels", default='./source/labels.pt', type=str)
parser.add_argument("path_target_embeddings", default='./target/train/embeddings.pt', type=str)
parser.add_argument("path_target_labels", default='./target/train/labels.pt', type=str)
parser.add_argument("path_target_val_embeddings", default='./target/val/embeddings.pt', type=str)
parser.add_argument("path_target_val_labels", default='./target/val/labels.pt', type=str)

# Parse the arguments
args = parser.parse_args()

def main(args):
    global training_iterations, modalities

    # device where everything is run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    # the models are wrapped into the ActionRecognition task which manages all the training steps
    classifier = DomainAdaptationNER(args)
    classifier.load_on_gpu(device)

    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        if args.resume_from is not None:
            classifier.load_last_model(args.resume_from)
        # define number of iterations I'll do with the actual batch: we do not reason with epochs but with iterations
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        training_iterations = args.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here

        #TODO: datasets for source and target
        train_source = EmbeddingDataset(args.path_source_embeddings, args.path_source_labels)
        train_target = EmbeddingDataset(args.path_target_embeddings, args.path_target_labels)
        val_target = EmbeddingDataset(args.path_target_val_embeddings, args.path_target_val_labels)

        #TODO: dataloaders for source and target
        train_loader_source = DataLoader(train_source)
        train_loader_target = DataLoader(train_loader_target)
        val_loader = DataLoader(val_target)
        train(classifier, train_loader_source, train_loader_target, val_loader, device)


    elif args.action == "validate":
        if args.resume_from is not None:
            classifier.load_last_model(args.resume_from)
        #TODO: val dataloader for source and target

        validate(classifier, val_loader, device, classifier.current_iter)


def train(classifier, train_loader_source, train_loader_target, val_loader, device):
    """
    function to train the model on the test set
    classifier: Task containing the model to be trained
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    num_classes: int, number of classes in the classification problem
    """
    global training_iterations, modalities

    data_loader_source = iter(train_loader_source)
    data_loader_target = iter(train_loader_source)
    classifier.train(True)
    classifier.zero_grad()
    iteration = classifier.current_iter * (args.total_batch // args.batch_size)

    # the batch size should be total_batch but batch accumulation is done with batch size = batch_size.
    # real_iter is the number of iterations if the batch size was really total_batch
    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args.lr_steps:
            # learning rate decay at iteration = lr_steps
            classifier.reduce_learning_rate()
        # gradient_accumulation_step is a bool used to understand if we accumulated at least total_batch
        # samples' gradient
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        # the following code is necessary as we do not reason in epochs so as soon as the dataloader is finished we need
        # to redefine the iterator
        try:
            source_data, source_label = next(data_loader_source)
            
        except StopIteration:
            data_loader_source = iter(train_loader_source)
            source_data, source_label = next(data_loader_source)
        
        try:
            target_data, target_label = next(data_loader_target)
            
        except StopIteration:
            data_loader_target= iter(train_loader_target)
            target_data, target_label = next(data_loader_target)

        source_label = source_label.to(device)
        target_label=target_label.to(device)
        
        data_source= {}
        data_target= {}
    
        data_source = source_data.to(device)
        data_target = target_data.to(device)


        if data_source is None or data_target is None :
            raise UserWarning('train_classifier: Cannot be None type')
        logits, features = classifier.forward(data_source, data_target)

        classifier.compute_loss(logits, source_label, features)
        classifier.backward(retain_graph=False)
        classifier.compute_accuracy(logits, source_label)

        # update weights and zero gradients if total_batch samples are passed
        if gradient_accumulation_step:
            classifier.check_grad()
            classifier.step()
            classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done, notice we validate and
        # save the last 9 models
        if gradient_accumulation_step and real_iter % args.eval_freq == 0:
            val_metrics = validate(classifier, val_loader, device, int(real_iter))

            if val_metrics['top1'] <= classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                classifier.best_iter = real_iter
                classifier.best_iter_score = val_metrics['top1']

            classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            classifier.train(True)


def validate(model, val_loader, device, it):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            # batch = data.shape[0]
            # logits = torch.zeros((batch, num_classes)).to(device)

            data = data.to(device)

            output, _ = model(data, is_train=False)

            model.compute_accuracy(output, label)

            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, f'val_precision.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main(args)
