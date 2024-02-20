import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.init import normal_, constant_
from collections import OrderedDict, defaultdict
import logging
from typing import Dict, Optional
from utils import logger
from utils import metrics
import os
from datetime import datetime
from pathlib import Path
from utils.logger import logger
from utils.args import writer
from collections import defaultdict

class AdaptiveModule(nn.Module):

    def __init__(self, in_features_dim, model_config, num_classes_source=None, num_classes_target=None):

        super(AdaptiveModule, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config

        self.multi_head_attention = nn.MultiheadAttention(in_features_dim, num_heads=8, dropout=model_config.dropout)
        self.fc_task_specific_layer = self.TaskModule(model_config.num_fcl, in_features_dim=in_features_dim, out_features_dim=in_features_dim, dropout=model_config.dropout)
        
        if 'token_domain_classifier' in self.model_config.blocks:
            self.token_domain_classifier = self.DomainClassifier(in_features_dim, model_config.beta_token)
        
        if 'window_domain_classifier' in self.model_config.blocks or 'game_module' in self.model_config.blocks:
            self.fc_window_features = self.FullyConnectedLayer(model_config.window_size * in_features_dim, in_features_dim)

        if 'window_domain_classifier' in self.model_config.blocks:
            self.window_domain_classifier = self.DomainClassifier(in_features_dim, model_config.beta_window)

        if 'game_module' in self.model_config.blocks:
            self.game_module_source = self.GameModule(in_features_dim, model_config.window_size, num_classes_source)
            self.game_module_target = self.GameModule(in_features_dim, model_config.window_size, num_classes_target)

        self.fc_classifier_source = nn.Linear(in_features_dim, num_classes_source)
        self.fc_classifier_target = nn.Linear(in_features_dim, num_classes_target)
        std = 0.001
        
        if num_classes_target is not None:
            normal_(self.fc_classifier_target.weight, 0, std)
            constant_(self.fc_classifier_target.bias, 0)

        if num_classes_source is not None:
            normal_(self.fc_classifier_source.weight, 0, std)
            constant_(self.fc_classifier_source.bias, 0)

    def forward(self, source=None, target=None, class_labels_source=None, class_labels_target=None, is_train=True):
        output = defaultdict(lambda: None)

        for domain, feats, class_labels in [('source', source, class_labels_source), ('target', target, class_labels_target)]:
            if feats is not None:
                # feats = self.multi_head_attention(feats, feats, feats)[0]
                feats = self.fc_task_specific_layer(feats)
                output[f'feats_fcl'] = feats
                if domain == 'source':
                    output[f'preds_class_{domain}'] = self.fc_classifier_source(feats)
                else:
                    output[f'preds_class_{domain}'] = self.fc_classifier_target(feats)
            else:
                continue

            if 'token_domain_classifier' in self.model_config.blocks and is_train:
                output[f'preds_domain_token_{domain}'] = self.token_domain_classifier(feats)

            if ('window_domain_classifier' in self.model_config.blocks or 'game_module' in self.model_config.blocks) and is_train:
                try:
                    window_class_labels = torch.vstack(tuple(torch.hstack(tuple(class_labels[i+start] if i+start<len(class_labels) else torch.zeros((1,)).to(self.device) for i in range(self.model_config.window_size))) for start in range(len(class_labels))))
                except:
                    raise Exception(f'Could not create window_class_labels_{domain}, class_labels.shape: {class_labels.shape}, self.model_config.window_size: {self.model_config.window_size}')
                
                feats_window = torch.vstack(tuple(torch.hstack(tuple(feats[i+start,:] if i+start<len(feats) else torch.zeros(feats.shape[1:]).to(self.device) for i in range(self.model_config.window_size))) for start in range(len(feats))))
                feats_window = self.fc_window_features(feats_window)

                if 'window_domain_classifier' in self.model_config.blocks:
                    output[f'preds_domain_window_{domain}'] = self.window_domain_classifier(feats_window)

                if 'game_module' in self.model_config.blocks:
                    if domain == 'source':
                        output[f'wordle_{domain}'] = self.game_module_source.play(feats_window, window_class_labels)
                    else:
                        output[f'wordle_{domain}'] = self.game_module_target.play(feats_window, window_class_labels)

                output[f'window_class_labels_{domain}'] = window_class_labels

        return output

    class TaskModule(nn.Module):
        def __init__(self, n_fcl, in_features_dim, out_features_dim, dropout=0.5):
            
            super(AdaptiveModule.TaskModule, self).__init__()
            
            fc_layers = []
            std = 0.001


            for i in range(n_fcl):
                
                if i == 0:
                    fcl = AdaptiveModule.FullyConnectedLayer(in_features_dim, out_features_dim, dropout)

                else:
                    fcl = AdaptiveModule.FullyConnectedLayer(out_features_dim, out_features_dim, dropout)
                normal_(fcl.weight, 0, std)
                constant_(fcl.bias, 0)

                fc_layers.append(fcl)
            
            self.fc_layers = nn.Sequential(*fc_layers)
        
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

    class GameModule(nn.Module):

        def __init__(self, in_features_dim, window_size, n_classes, dropout=0.5, n_attempts=6) -> None:
            
            super(AdaptiveModule.GameModule, self).__init__()

            self.window_size = window_size
            self.n_classes = n_classes
            
            self.fc_layer = AdaptiveModule.FullyConnectedLayer(in_features_dim+2*window_size, window_size*n_classes, dropout)
            self.softmax = torch.nn.Softmax(dim=2)
            self.n_attempts = n_attempts
            
        
        def play(self, feats, gt):

            """
            Attempts the wordle game for n_attempts times,
            returns the last guess
            """
            
            hint = torch.zeros_like(gt)
            last_attempt = torch.zeros_like(gt)

            for _ in range(self.n_attempts):
                logits = self.forward(feats, hint, last_attempt)
                last_attempt = torch.argmax(logits, dim=2)
                try:
                    hint = last_attempt == gt
                except:
                    raise Exception(f'Could not make hint, gt shape: {gt.shape}, last_attempt shape: {last_attempt.shape}')
            
            return logits
        
        def forward(self, feats, hint, last_attempt):

            """
            How the game works:
            - feats: features of the window
            - hint: 0 if the entity is not in the window, 1 if the entity is in the window but not in the right position, 2 if the entity is in the window and in the right position
            - last_attempt: last attempt of the player
            A fully connected layer is used to predict the next attempt, that is the joint distribution of the entities and the position
            The dimension of feats is (number of windows in the batch, window_size, entity classes)
            """
            try:
                feats = torch.cat((feats, hint, last_attempt), dim=1)
            except:
                raise Exception(f'Failed concatenating, shape of feats: {feats.shape}, shape of hint: {hint.shape}, shape of last_attempt: {last_attempt.shape}\n\
                                Last attempt was {last_attempt}')
            feats = self.fc_layer(feats)
            feats = feats.view((-1,self.window_size, self.n_classes))
            logits = self.softmax(feats)
            return logits


class DomainAdaptationNER(nn.Module):

    def __init__(self, args) -> None:

        super(DomainAdaptationNER, self).__init__()
        
        self.args = args

        self.name = self.args.name
        self.models_dir = self.args.models_dir

        self.best_iter_score = 0
        self.model_count = 0

        self.total_batch = args.total_batch
        self.batch_size = args.batch_size

        self.current_iter = 0

        args.blocks = []

        if not args.remove_window_domain_classifier:
            args.blocks.append('window_domain_classifier')
        if not args.remove_token_domain_classifier:
            args.blocks.append('token_domain_classifier')
        if not args.remove_wordle_game_module:
            args.blocks.append('game_module')
        
        logger.info(f'Blocks: {args.blocks}')

        self.blocks = args.blocks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AdaptiveModule(args.in_features_dim, args, args.num_classes_target, args.num_classes_source)
        self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                    reduce=None, reduction='none')
        
        self.token_domain_classifier_params = self.window_domain_classifier_params = []
        self.token_domain_classifier_params_ = self.window_domain_classifier_params_ = []
        if 'token_domain_classifier' in self.blocks:
            self.token_domain_classifier_params = list(p.data_ptr() for p in self.model.token_domain_classifier.parameters())
            self.token_domain_classifier_params_ = filter(lambda p: p.requires_grad, self.model.token_domain_classifier.parameters())
        
        if 'window_domain_classifier' in self.blocks:
            self.window_domain_classifier_params = list(p.data_ptr() for p in self.model.window_domain_classifier.parameters())
            self.window_domain_classifier_params_ = filter(lambda p: p.requires_grad, self.model.window_domain_classifier.parameters())

        self.optim_params = filter(lambda p: p.requires_grad and not (p.data_ptr() in self.window_domain_classifier_params or p.data_ptr() in self.token_domain_classifier_params), self.model.parameters())

        self.optimizer = torch.optim.SGD([{'params': self.optim_params, 'lr': args.lr},
                                            {'params': self.window_domain_classifier_params_, 'lr': args.lr_discriminator},
                                            {'params': self.token_domain_classifier_params_, 'lr': args.lr_discriminator}],
                                            weight_decay=args.weight_decay,
                                            momentum=args.sgd_momentum)
        
        self.accuracy = {}
        self.accuracy['source'] = metrics.Accuracy(topk=(1,), classes=args.num_classes_source)
        self.accuracy['target'] = metrics.Accuracy(topk=(1,), classes=args.num_classes_target)

        self.f1 = {}
        self.f1['source'] = metrics.F1(topk=(1,), classes=args.num_classes_source)
        self.f1['target'] = metrics.F1(topk=(1,), classes=args.num_classes_target)

        self.num_classes_source = args.num_classes_source
        self.num_classes_target = args.num_classes_target

        self.domain_token_loss = metrics.AverageMeter()
        self.domain_window_loss = metrics.AverageMeter()
        self.classification_loss_source = metrics.AverageMeter()
        self.classification_loss_target = metrics.AverageMeter()

        self.wordle_source_position_loss = metrics.AverageMeter()
        self.wordle_target_position_loss = metrics.AverageMeter()

        self.wordle_source_window_loss = metrics.AverageMeter()
        self.wordle_target_window_loss = metrics.AverageMeter()
    
    def forward(self, source = None, target = None, class_labels_source: 'torch.Tensor' = None, class_labels_target: 'torch.Tensor' = None, is_train=True):
        return self.model(source, target, class_labels_source, class_labels_target, is_train=is_train)

    def compute_loss(self, class_labels_source: 'torch.Tensor', class_labels_target: 'torch.Tensor', predictions: Dict[str, 'torch.Tensor']):
        
        try:
            classification_loss_source = self.criterion(predictions['preds_class_source'], class_labels_source) #cross entropy loss
            classification_loss_target = self.criterion(predictions['preds_class_target'], class_labels_target) #cross entropy loss
        except:
            raise ValueError(f'Could not compute classification loss, predictions: {predictions["preds_class_source"].shape}', f'Class labels source: {class_labels_source.shape}',\
                             f'Class labels target: {class_labels_target.shape}, predictions: {predictions["preds_class_target"].shape}', \
                             f'Labels source: {class_labels_source.unique()}', f'Labels target: {class_labels_target.unique()}')

        self.classification_loss_source.update(torch.mean(classification_loss_source) / (self.total_batch / self.batch_size), self.batch_size)
        self.classification_loss_target.update(torch.mean(classification_loss_target) / (self.total_batch / self.batch_size), self.batch_size)
        
        if 'token_domain_classifier' in self.blocks:
            preds_domain_token_source = predictions['preds_domain_token_source']
            domain_label_source=torch.zeros(preds_domain_token_source.shape[0], dtype=torch.int64)    
            
            preds_domain_token_target = predictions['preds_domain_token_target']
            domain_label_target=torch.ones(preds_domain_token_target.shape[0], dtype=torch.int64)    

            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_domain_token_all=torch.cat((preds_domain_token_source, preds_domain_token_target),0)

            domain_token_loss = self.criterion(pred_domain_token_all, domain_label_all)
            self.domain_token_loss.update(torch.mean(domain_token_loss) / (self.total_batch / self.batch_size), self.batch_size)

        if 'window_domain_classifier' in self.blocks:
            preds_domain_window_source = predictions['preds_domain_window_source']
            domain_label_source=torch.zeros(preds_domain_window_source.shape[0], dtype=torch.int64)    
            
            preds_domain_window_target = predictions['preds_domain_window_target']
            domain_label_target=torch.ones(preds_domain_window_target.shape[0], dtype=torch.int64)    

            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_domain_window_all=torch.cat((preds_domain_window_source, preds_domain_window_target),0)

            domain_window_loss = self.criterion(pred_domain_window_all, domain_label_all)
            self.domain_window_loss.update(torch.mean(domain_window_loss) / (self.total_batch / self.batch_size), self.batch_size)
        
        if 'game_module' in self.blocks:
            wordle_source = predictions['wordle_source'] # Dimension: (windows_in_batch, window_size, num_classes)
            wordle_target = predictions['wordle_target']

            window_class_labels_source = predictions['window_class_labels_source'] # Dimension: (windows_in_batch, window_size)
            window_class_labels_target = predictions['window_class_labels_target']

            # Now we need something like (windows_in_batch*window_size, num_classes)
            # We do this by one-hot encoding the labels
            window_class_labels_source_one_hot = torch.nn.functional.one_hot(window_class_labels_source.long(), num_classes=self.num_classes_source).to(float)
            window_class_labels_target_one_hot = torch.nn.functional.one_hot(window_class_labels_target.long(), num_classes=self.num_classes_target).to(float)

            window_class_labels_source_one_hot_word = window_class_labels_source_one_hot.view(-1, self.num_classes_source)
            window_class_labels_target_one_hot_word = window_class_labels_target_one_hot.view(-1, self.num_classes_target)

            wordle_source_position = wordle_source.view(-1, self.num_classes_source)
            wordle_target_position = wordle_target.view(-1, self.num_classes_target)

            wordle_source_position_loss = self.criterion(wordle_source_position, window_class_labels_source_one_hot_word)
            wordle_target_position_loss = self.criterion(wordle_target_position, window_class_labels_target_one_hot_word)

            wordle_source_position_loss = torch.mean(wordle_source_position_loss) / (self.total_batch / self.batch_size)
            wordle_target_position_loss = torch.mean(wordle_target_position_loss) / (self.total_batch / self.batch_size)

            self.wordle_source_position_loss.update(wordle_source_position_loss, self.batch_size)
            self.wordle_target_position_loss.update(wordle_target_position_loss, self.batch_size)

            # Now we need to compute the loss which does not take into account the position of the entity

            wordle_source_window = torch.ones((wordle_source.shape[0], wordle_source.shape[2]))-torch.prod(torch.ones_like(wordle_source)-wordle_source, dim=1).to(self.device) # Dimension: (windows_in_batch, num_classes)
            wordle_target_window = torch.ones((wordle_target.shape[0], wordle_target.shape[2]))-torch.prod(torch.ones_like(wordle_target)-wordle_target, dim=1).to(self.device)

            window_class_labels_source_one_hot_window = window_class_labels_source_one_hot.sum(dim=1)
            window_class_labels_target_one_hot_window = window_class_labels_target_one_hot.sum(dim=1)

            try:
                wordle_source_window_loss = self.criterion(wordle_source_window, window_class_labels_source_one_hot_window)
                wordle_target_window_loss = self.criterion(wordle_target_window, window_class_labels_target_one_hot_window)
            except:
                raise Exception(f'Pred shape: {wordle_source_window.shape}', f'Label shape: {window_class_labels_source_one_hot.shape}')

            wordle_source_window_loss = torch.mean(wordle_source_window_loss) / (self.total_batch / self.batch_size)
            wordle_target_window_loss = torch.mean(wordle_target_window_loss) / (self.total_batch / self.batch_size)

            self.wordle_source_window_loss.update(wordle_source_window_loss, self.batch_size)
            self.wordle_target_window_loss.update(wordle_target_window_loss, self.batch_size)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10

        """new_lr = self.optimizer.param_groups[-1]["lr"] / 10
        self.optimizer.param_groups[-1]["lr"] = new_lr"""

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        
        #TODO: reset game module loss

        if 'token_domain_classifier' in self.blocks:
            self.domain_token_loss.reset()
        
        if 'window_domain_classifier' in self.blocks:
            self.domain_window_loss.reset()
        
        if 'game_module' in self.blocks:
            self.wordle_source_position_loss.reset()
            self.wordle_target_position_loss.reset()
            self.wordle_source_window_loss.reset()
            self.wordle_target_window_loss.reset()

        self.classification_loss_source.reset()
        self.classification_loss_target.reset()
    
    def compute_accuracy(self, output: Dict[str, 'torch.Tensor'], class_labels_source: 'torch.Tensor' = None, class_labels_target: 'torch.Tensor' = None):
        """Compute the classification accuracy for source and target.

        Parameters
        ----------
        output : Dict[str, torch.Tensor]
            output of the model
        label : torch.Tensor
            ground truth
        """
        if class_labels_source is not None:
            self.accuracy['source'].update(output['preds_class_source'], class_labels_source)
        if class_labels_target is not None:
            self.accuracy['target'].update(output['preds_class_target'], class_labels_target)

    def compute_f1(self, output: 'torch.Tensor', class_labels: 'torch.Tensor', domain: 'str'):
        """Compute the classification accuracy for source and target.

        Parameters
        ----------
        output : Dict[str, torch.Tensor]
            output of the model
        label : torch.Tensor
            ground truth
        """
        
        self.f1[domain].update(output, class_labels)
        

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy['source'].reset()
        self.accuracy['target'].reset()
        self.f1['source'].reset()
        self.f1['target'].reset()


    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            if self.args.action != "gridsearch":
                writer.add_scalar(f'learning_rates/lr_{i}', param_group["lr"], global_step=int(self.current_iter))
        self.optimizer.step()
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
        
        if 'token_domain_classifier' in self.blocks:
            loss += self.domain_token_loss.val
        
        if 'window_domain_classifier' in self.blocks:
            loss += self.domain_window_loss.val
        
        if 'game_module' in self.blocks:
            wordle_loss = self.wordle_source_position_loss.val
            wordle_loss += self.wordle_target_position_loss.val
            wordle_loss += self.wordle_source_window_loss.val
            wordle_loss += self.wordle_target_window_loss.val
            loss += self.args.beta_wordle*wordle_loss

        loss.backward(retain_graph=retain_graph)
    
    def load_on_gpu(self, device: torch.device = torch.device("cuda")):
        """Load all the models on the GPU(s) using DataParallel.

        Parameters
        ----------
        device : torch.device, optional
            the device to move the models on, by default torch.device('cuda')
        """

        self.model = torch.nn.DataParallel(self.model).to(device)
    
    def check_grad(self):
        """Check that the gradients of the model are not over a certain threshold."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.norm(2).item() > 25:
                    logger.info(f"Param {name} has a gradient whose L2 norm is over 25")

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
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # Restore the optimizer parameters
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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
        # build the filename of the model
        if prefix is not None:
            filename = prefix + "_" + self.name + "_" + str(self.model_count) + ".pth"
        else:
            filename = self.name + "_" + str(self.model_count) + ".pth"

        if not os.path.exists(os.path.join(self.models_dir, self.args.experiment_dir)):
            os.makedirs(os.path.join(self.models_dir, self.args.experiment_dir))

        try:
            torch.save(
                {
                    "iteration": current_iter,
                    "best_iter": self.best_iter,
                    "best_iter_score": self.best_iter_score,
                    "acc_mean": last_iter_acc,
                    "loss_cls_source_mean": self.classification_loss_source.acc,
                    "loss_cls_target_mean": self.classification_loss_target.acc,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
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
        self.model.train(mode)
        if self.model.module.fc_task_specific_layer.training != mode:
            raise Exception(f'Error: model is in {self.model.training} mode, but fc_task_specific_layer is in {self.fc_task_specific_layer.training} mode')
