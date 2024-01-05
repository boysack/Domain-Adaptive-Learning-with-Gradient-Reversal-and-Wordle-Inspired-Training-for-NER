import utils
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.init import normal_, constant_
from collections import OrderedDict
import logging
from typing import Dict, Optional
from utils.logger import logger
import os
from datetime import datetime
from pathlib import Path

class AdaptiveModule(nn.Module):

    VALID_ENDPOINTS = (
        'Backbone',
        'Spatial module',
        'Temporal module',
        'Gy',
        'Logits',
        'Predictions',
    )

    def __init__(self, in_features_dim, model_config, num_classes_target, num_classes_source=None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        super(AdaptiveModule, self).__init__()
       
        self.fc_task_specific_layer = self.TaskModule(in_features_dim=in_features_dim, out_features_dim=in_features_dim, dropout=model_config.dropout)
        
        if 'token_domain_classifier' in self.model_config.blocks:
            self.token_domain_classifier = self.DomainClassifier(in_features_dim, model_config.beta0)
        
        if 'window_domain_classifier' in self.model_config.blocks:
            self.fc_window_features = self.FullyConnectedLayer(model_config.window_size * in_features_dim, in_features_dim)
            self.window_domain_classifier = self.DomainClassifier(in_features_dim, model_config.beta1)

        self.fc_classifier_source = nn.Linear(in_features_dim, num_classes_source)
        self.fc_classifier_target = nn.Linear(in_features_dim, num_classes_target)
        std = 0.001
        
        normal_(self.fc_classifier_target.weight, 0, std)
        constant_(self.fc_classifier_target.bias, 0)

        if num_classes_source is not None:
            normal_(self.fc_classifier_source.weight, 0, std)
            constant_(self.fc_classifier_source.bias, 0)

    def forward(self, source, target, is_train=True):
        feats_source = self.fc_task_specific_layer(source)
        feats_target = self.fc_task_specific_layer(target)

        if 'token_domain_classifier' in self.model_config.blocks:
            preds_domain_token_source = self.token_domain_classifier(feats_source)
            preds_domain_token_target = self.token_domain_classifier(feats_target)
        
        if 'window_domain_classifier' in self.model_config.blocks:
            
            feats_window_source = self.fc_window_features(feats_source)
            feats_window_target = self.fc_window_features(feats_target)

            preds_domain_window_source = self.window_domain_classifier(feats_window_source)
            preds_domain_window_target = self.window_domain_classifier(feats_window_target)

        preds_class_source = self.fc_classifier_source(feats_source)
        preds_class_target = self.fc_classifier_target(feats_target)

        return {'preds_class_source': preds_class_source, 'preds_class_target': preds_class_target,\
                'preds_domain_token_source': preds_domain_token_source, 'preds_domain_token_target': preds_domain_token_target,\
                'preds_domain_window_source': preds_domain_window_source, 'preds_domain_window_target': preds_domain_window_target}

    class TaskModule(nn.Module):
        def __init__(self, n_fcl, in_features_dim, out_features_dim, dropout=0.5):
            
            super(AdaptiveModule.TaskModule, self).__init__()
            
            fc_layers = []
            fc_layers.append(AdaptiveModule.FullyConnectedLayer(in_features_dim, out_features_dim, dropout))

            for i in range(n_fcl-1):
                fc_layers.append(AdaptiveModule.FullyConnectedLayer(out_features_dim, out_features_dim, dropout))
            
            self.fc_layers = nn.Sequential(fc_layers)
            
            self.bias = self.fc_layers.bias
            self.weight = self.fc_layers.weight

            std = 0.001
            normal_(self.weight, 0, std)
            constant_(self.bias, 0)
        
        def forward(self, x):
            return self.fc_layers(x)
    
    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features_dim, out_features_dim, dropout=0.5):
            super(AdaptiveModule.FullyConnectedLayer, self).__init__()
            self.in_features_dim = in_features_dim
            self.out_features_dim = out_features_dim
            
            """Here I am doing what is done in the official code, 
            in the first fc layer the output dimension is the minimum between the input feature dimension and 1024"""
            self.relu = nn.ReLU() # Again using the architecture of the official code
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(self.in_features_dim, self.out_features_dim)
            self.bias = self.fc.bias
            self.weight = self.fc.weight
            std = 0.001
            normal_(self.fc.weight, 0, std)
            constant_(self.fc.bias, 0)
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x               

    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, beta):
            ctx.beta = beta
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.neg() * ctx.beta
            return grad_input, None
    
    class DomainClassifier(nn.Module):

        def __init__(self, in_features_dim, beta):

            std = 0.001

            super(AdaptiveModule.DomainClassifier, self).__init__()
            self.in_features_dim = in_features_dim
            self.domain_classifier = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(self.in_features_dim, self.in_features_dim)),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(self.in_features_dim, 2))
            ]))
            self.beta = beta

            self.bias = nn.ParameterList([self.domain_classifier[0].bias, self.domain_classifier[2].bias])
            
            self.weight = nn.ParameterList([self.domain_classifier[0].weight, self.domain_classifier[2].weight])

            for bias in self.bias:
                constant_(bias, 0)
            
            for weight in self.weight:
                normal_(weight, 0, std)
                    
        def forward(self, x):
            x = AdaptiveModule.GradReverse.apply(x,self.beta)
            x = self.domain_classifier(x)
            return x


class DomainAdaptationNER(nn.Module):

    def __init__(self, args) -> None:
        
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AdaptiveModule(args.in_features_dim, args, args.num_classes_target, args.num_classes_source).to(self.device).train()

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                    reduce=None, reduction='none')
        
        self.optim_params = filter(lambda parameter: parameter.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.SGD(self.optim_params, args.lr,
                                            weight_decay=args.weight_decay,
                                            momentum=args.sgd_momentum)
        
        self.accuracy_source = utils.Accuracy(topk=(1,), classes=args.num_classes_source)
        self.accuracy_target = utils.Accuracy(topk=(1,), classes=args.num_classes_target)

        self.domain_token_loss = utils.AverageMeter()
        self.domain_window_loss = utils.AverageMeter()
        self.classification_loss_source = utils.AverageMeter()
        self.classification_loss_target = utils.AverageMeter()
    
    def forward(self, source, target, is_train=True):
        return self.model(source, target, is_train=is_train)

    def compute_loss(self, class_labels_source: 'torch.Tensor', class_labels_target: 'torch.Tensor', predictions: Dict[str, 'torch.Tensor']):
        classification_loss_source = self.criterion(predictions['preds_class_source'], class_labels_source) #cross entropy loss
        classification_loss_target = self.criterion(predictions['preds_class_target'], class_labels_target) #cross entropy loss

        self.classification_loss_source.update(torch.mean(classification_loss_source) / (self.total_batch / self.batch_size), self.batch_size)
        self.classification_loss_target.update(torch.mean(classification_loss_target) / (self.total_batch / self.batch_size), self.batch_size)
        
        if 'token_domain_classifier' in self.model_args.blocks:
            preds_domain_token_source = predictions['preds_domain_token_source']
            domain_label_source=torch.zeros(preds_domain_token_source.shape[0], dtype=torch.int64)    
            
            preds_domain_token_target = predictions['preds_domain_token_target']
            domain_label_target=torch.zeros(preds_domain_token_target.shape[0], dtype=torch.int64)    

            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_domain_token_all=torch.cat((preds_domain_token_source, domain_label_source),0)

            domain_token_loss = self.criterion(pred_domain_token_all, domain_label_all)
            self.domain_token_loss.update(torch.mean(domain_token_loss) / (self.total_batch / self.batch_size), self.batch_size)
        
        if 'window_domain_classifier' in self.model_args.blocks:
            preds_domain_window_source = predictions['preds_domain_window_source']
            domain_label_source=torch.zeros(preds_domain_window_source.shape[0], dtype=torch.int64)    
            
            preds_domain_window_target = predictions['preds_domain_window_target']
            domain_label_target=torch.zeros(preds_domain_window_target.shape[0], dtype=torch.int64)    

            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_domain_window_all=torch.cat((preds_domain_window_source, domain_label_source),0)

            domain_window_loss = self.criterion(pred_domain_window_all, domain_label_all)
            self.domain_window_loss.update(torch.mean(domain_window_loss) / (self.total_batch / self.batch_size), self.batch_size)

    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        # fused_logits = reduce(lambda x, y: x + y, logits.values())

        self.accuracy.update(logits, label)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        new_lr = self.optimizer.param_groups[-1]["lr"] / 10
        self.optimizer.param_groups[-1]["lr"] = new_lr

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        
        if 'token_domain_classifier' in self.model_args.blocks:
            self.domain_token_loss.reset()
        
        if 'window_domain_classifier' in self.model_args.blocks:
            self.domain_window_loss.reset()

        self.classification_loss_source.reset()
        self.classification_loss_target.reset()
    
    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        # fused_logits = reduce(lambda x, y: x + y, logits.values())

        self.accuracy_source.update(logits['preds_class_source'], label)
        self.accuracy_target.update(logits['preds_class_target'], label)

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy_source.reset()
        self.accuracy_target.update()


    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """

        loss = 0

        loss += self.classification_loss_source.val
        loss += self.classification_loss_target.val
        
        if 'token_domain_classifier' in self.model_args.blocks:
            loss += self.domain_token_loss.val
        
        if 'window_domain_classifier' in self.model_args.blocks:
            loss += self.domain_window_loss.val
        
        loss.backward(retain_graph=retain_graph)
    
    def load_on_gpu(self, device: torch.device = torch.device("cuda")):
        """Load all the models on the GPU(s) using DataParallel.

        Parameters
        ----------
        device : torch.device, optional
            the device to move the models on, by default torch.device('cuda')
        """
        for modality, model in self.task_models.items():
            self.task_models[modality] = torch.nn.DataParallel(model).to(device)

    def __restore_checkpoint(self, m: str, path: str):
        """Restore a checkpoint from path.

        Parameters
        ----------
        m : str
            modality to load from
        path : str
            path to load from
        """
        logger.info("Restoring {} for modality {} from {}".format(self.name, m, path))

        checkpoint = torch.load(path)

        # Restore the state of the task
        self.current_iter = checkpoint["iteration"]
        self.best_iter = checkpoint["best_iter"]
        self.best_iter_score = checkpoint["best_iter_score"]
        self.last_iter_acc = checkpoint["acc_mean"]

        # Restore the model parameters
        self.task_models[m].load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Restore the optimizer parameters
        self.optimizer[m].load_state_dict(checkpoint["optimizer_state_dict"])

        try:
            self.model_count = checkpoint["last_model_count_saved"]
            self.model_count = self.model_count + 1 if self.model_count < 9 else 1
        except KeyError:
            # for compatibility with models saved before refactoring
            self.model_count = 1

        logger.info(
            f"{m}-Model for {self.name} restored at iter {self.current_iter}\n"
            f"Best accuracy on val: {self.best_iter_score:.2f} at iter {self.best_iter}\n"
            f"Last accuracy on val: {self.last_iter_acc:.2f}\n"
            f"Last loss: {checkpoint['loss_mean']:.2f}"
        )

    def load_model(self, path: str, idx: int):
        """Load a specific model (idx-one) among the last 9 saved.

        Load a specific model (idx-one) among the last 9 saved from a specific path,
        might be overwritten in case the task requires it.

        Parameters
        ----------
        path : str
            directory to load models from
        idx : int
            index of the model to load
        """
        # List all the files in the path in chronological order (1st is most recent, last is less recent)
        last_dir = Path(
            list(
                sorted(
                    Path(path).iterdir(),
                    key=lambda date: datetime.strptime(os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S"),
                )
            )[-1]
        )
        last_models_dir = last_dir.iterdir()

        for m in self.modalities:
            # Get the correct model (modality, name, idx)
            model = list(
                filter(
                    lambda x: m == x.name.split(".")[0].split("_")[-2]
                    and self.name == x.name.split(".")[0].split("_")[-3]
                    and str(idx) == x.name.split(".")[0].split("_")[-1],
                    last_models_dir,
                )
            )[0].name
            model_path = os.path.join(str(last_dir), model)

            self.__restore_checkpoint(m, model_path)

    def load_last_model(self, path: str):
        """Load the last model from a specific path.

        Parameters
        ----------
        path : str
            directory to load models from
        """
        # List all the files in the path in chronological order (1st is most recent, last is less recent)
        last_models_dir = list(
            sorted(
                Path(path).iterdir(),
                key=lambda date: datetime.strptime(os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S"),
            )
        )[-1]
        saved_models = [x for x in reversed(sorted(Path(last_models_dir).iterdir(), key=os.path.getmtime))]

        for m in self.modalities:
            # Get the correct model (modality, name, idx)
            model = list(
                filter(
                    lambda x: m == x.name.split(".")[0].split("_")[-2]
                    and self.name == x.name.split(".")[0].split("_")[-3],
                    saved_models,
                )
            )[0].name

            model_path = os.path.join(last_models_dir, model)
            self.__restore_checkpoint(m, model_path)

    def save_model(self, current_iter: int, last_iter_acc: float, prefix: Optional[str] = None):
        """Save the model.

        Parameters
        ----------
        current_iter : int
            current iteration in which the model is going to be saved
        last_iter_acc : float
            accuracy reached in the last iteration
        prefix : Optional[str], optional
            string to be put as a prefix to filename of the model to be saved, by default None
        """
        for m in self.modalities:
            # build the filename of the model
            if prefix is not None:
                filename = prefix + "_" + self.name + "_" + m + "_" + str(self.model_count) + ".pth"
            else:
                filename = self.name + "_" + m + "_" + str(self.model_count) + ".pth"

            if not os.path.exists(os.path.join(self.models_dir, self.args.experiment_dir)):
                os.makedirs(os.path.join(self.models_dir, self.args.experiment_dir))

            try:
                torch.save(
                    {
                        "iteration": current_iter,
                        "best_iter": self.best_iter,
                        "best_iter_score": self.best_iter_score,
                        "acc_mean": last_iter_acc,
                        "loss_mean": self.classification_loss.acc,
                        "model_state_dict": self.task_models[m].state_dict(),
                        "optimizer_state_dict": self.optimizer[m].state_dict(),
                        "last_model_count_saved": self.model_count,
                    },
                    os.path.join(self.models_dir, self.args.experiment_dir, filename),
                )
                self.model_count = self.model_count + 1 if self.model_count < 9 else 1

            except Exception as e:
                logger.error("An error occurred while saving the checkpoint: ")
                logger.error(e)

    def train(self, mode: bool = True):
        """Activate the training in all models.

        Activate the training in all models (when training, DropOut is active, BatchNorm updates itself)
        (when not training, BatchNorm is freezed, DropOut disabled).

        Parameters
        ----------
        mode : bool, optional
            train mode, by default True
        """
        for model in self.task_models.values():
            model.train(mode)
