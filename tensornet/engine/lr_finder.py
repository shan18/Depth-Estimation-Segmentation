# Some snippets for the code in this file are referenced from
# https://github.com/davidtvs/pytorch-lr-finder


import os
import copy
import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

from tensornet.engine.learner import Learner
from tensornet.utils.progress_bar import ProgressBar
from tensornet.data.processing import InfiniteDataLoader


class LRFinder:
    """Learning rate range test.
    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.

    Args:
        model (torch.nn.Module): Model Instance.
        optimizer (torch.optim): Optimizer where the defined learning
            is assumed to be the lower boundary of the range test.
        criterion (torch.nn): Loss function.
        metric (str, optional): Metric to use for finding the best learning rate. Can
            be either 'loss' or 'accuracy'. (default: 'loss')
        device (str or torch.device, optional): Device where the computation
            will take place. If None, uses the same device as `model`. (default: none)
        memory_cache (bool, optional): If this flag is set to True, state_dict of
            model and optimizer will be cached in memory. Otherwise, they will be saved
            to files under the `cache_dir`. (default: True)
        cache_dir (str, optional): Path for storing temporary files. If no path is
            specified, system-wide temporary directory is used. Notice that this
            parameter will be ignored if `memory_cache` is True. (default: None)
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        metric='loss',
        device=None,
        memory_cache=True,
        cache_dir=None,
    ):
        # Parameter validation

        # Check if correct 'metric' has been given
        if not metric in ['loss', 'accuracy']:
            raise ValueError(f'For "metric" expected one of (loss, accuracy), got {metric}')

        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self._check_for_scheduler()

        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.history = {'lr': [], 'metric': []}
        self.best_metric = None
        self.best_lr = None
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir
        self.learner = None

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(memory_cache, cache_dir=cache_dir)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        self.device = self.model_device if not device else device

    def reset(self):
        """Restores the model and optimizer to their initial states."""
        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

        if not self.learner is None:
            self.learner.reset_history()

    def _check_for_scheduler(self):
        """Check if the optimizer has and existing scheduler attached to it."""
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                raise RuntimeError('Optimizer already has a scheduler attached to it')
    
    def _set_learning_rate(self, new_lrs):
        """Set the given learning rates in the optimizer."""
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                'Length of new_lrs is not equal to the number of parameter groups in the given optimizer'
            )

        # Set the learning rates to the parameter groups
        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = new_lr

    def range_test(
        self,
        train_loader,
        iterations,
        mode='iteration',
        val_loader=None,
        start_lr=None,
        end_lr=10,
        step_mode='exp',
        smooth_f=0.0,
        diverge_th=5,
    ):
        """Performs the learning rate range test.

        Args:
            train_loader (torch.utils.data.DataLoader): The training set data loader.
            iterations (int): The number of iterations/epochs over which the test occurs.
                If 'mode' is set to 'iteration' then it will correspond to the
                number of iterations else if mode is set to 'epoch' then it will correspond
                to the number of epochs.
            mode (str, optional): After which mode to update the learning rate. Can be
                either 'iteration' or 'epoch'. (default: 'iteration') 
            val_loader (torch.utils.data.DataLoader, optional): If None, the range test
                will only use the training metric. When given a data loader, the model is
                evaluated after each iteration on that dataset and the evaluation metric
                is used. Note that in this mode the test takes significantly longer but
                generally produces more precise results. (default: None)
            start_lr (float, optional): The starting learning rate for the range test.
                If None, uses the learning rate from the optimizer. (default: None)
            end_lr (float, optional): The maximum learning rate to test. (default: 10)
            step_mode (str, optional): One of the available learning rate policies,
                linear or exponential ('linear', 'exp'). (default: 'exp')
            smooth_f (float, optional): The metric smoothing factor within the [0, 1]
                interval. Disabled if set to 0, otherwise the metric is smoothed using
                exponential smoothing. (default: 0.0)
            diverge_th (int, optional): The test is stopped when the metric surpasses the
                threshold: diverge_th * best_metric. To disable, set it to 0. (default: 5)
        """

        # Check if correct 'mode' mode has been given
        if not mode in ['iteration', 'epoch']:
            raise ValueError(f'For "mode" expected one of (iteration, epoch), got {mode}')

        # Reset test results
        self.history = {'lr': [], 'metric': []}
        self.best_metric = None
        self.best_lr = None

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Initialize the proper learning rate policy
        if step_mode.lower() == 'exp':
            lr_schedule = ExponentialLR(self.optimizer, end_lr, iterations)
        elif step_mode.lower() == 'linear':
            lr_schedule = LinearLR(self.optimizer, end_lr, iterations)
        else:
            raise ValueError(f'Expected one of (exp, linear), got {step_mode}')

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError('smooth_f is outside the range [0, 1]')

        # Get the learner object
        self.learner = Learner(
            self.model, self.optimizer, self.criterion, train_loader,
            device=self.device, val_loader=val_loader
        )

        train_iterator = InfiniteDataLoader(train_loader)
        pbar = ProgressBar(target=iterations, width=8)
        if mode == 'iteration':
            print(mode.title() + 's')
        for iteration in range(iterations):
            # Train model
            if mode == 'epoch':
                print(f'{mode.title()} {iteration + 1}:')
            self._train_model(mode, train_iterator)
            if val_loader:
                self.learner.validate(verbose=False)
            
            # Get metric value
            metric_value = self._get_metric(val_loader)

            # Update the learning rate
            lr_schedule.step()
            self.history['lr'].append(lr_schedule.get_lr()[0])

            # Track the best metric and smooth it if smooth_f is specified
            if iteration == 0:
                self.best_metric = metric_value
                self.best_lr = self.history['lr'][-1]
            else:
                if smooth_f > 0:
                    metric_value = smooth_f * metric_value + (1 - smooth_f) * self.history['metric'][-1]
                if (
                    (self.metric == 'loss' and metric_value < self.best_metric) or
                    (self.metric == 'accuracy' and metric_value > self.best_metric)
                ):
                    self.best_metric = metric_value
                    self.best_lr = self.history['lr'][-1]

            # Check if the metric has diverged; if it has, stop the test
            self.history['metric'].append(metric_value)
            metric_value = self._display_metric_value(metric_value)
            if (
                diverge_th > 0 and
                ((self.metric == 'loss' and metric_value > self.best_metric * diverge_th) or
                (self.metric == 'accuracy' and metric_value < self.best_metric / diverge_th))
            ):
                if mode == 'iteration':
                    pbar.update(iterations - 1, values=[
                        ('lr', self.history['lr'][-1]),
                        (self.metric.title(), metric_value)
                    ])
                print('\nStopping early, the loss has diverged.')
                break
            else:
                if mode == 'epoch':
                    lr = self.history['lr'][-1]
                    print(f'Learning Rate: {lr:.4f}, {self.metric.title()}: {metric_value:.2f}\n')
                elif mode == 'iteration':
                    pbar.update(iteration, values=[
                        ('lr', self.history['lr'][-1]),
                        (self.metric.title(), metric_value)
                    ])
        
        metric = self._display_metric_value(self.best_metric)
        if mode == 'epoch':
            print(f'Learning Rate: {self.best_lr:.4f}, {self.metric.title()}: {metric:.2f}\n')
        elif mode == 'iteration':
            pbar.add(1, values=[
                ('lr', self.best_lr),
                (self.metric.title(), metric)
            ])
        print('Learning rate search finished.')
    
    def _train_model(self, mode, train_iterator):
        if mode == 'iteration':
            self.learner.model.train()
            data, targets = train_iterator.get_batch()
            loss = self.learner.train_batch(data, targets)
            accuracy = 100 * self.learner.train_correct / self.learner.train_processed
            self.learner.update_training_history(loss, accuracy)
        elif mode == 'epoch':
            self.learner.train_epoch()
    
    def _get_metric(self, validation=None):
        if self.metric == 'loss':
            if validation:
                return self.learner.val_losses[-1]
            return self.learner.train_losses[-1]
        elif self.metric == 'accuracy':
            if validation:
                return self.learner.val_accuracies[-1] / 100
            return self.learner.train_accuracies[-1] / 100
    
    def _display_metric_value(self, value):
        if self.metric == 'accuracy':
            return value * 100
        return value

    def plot(self, log_lr=True, show_lr=None):
        """Plots the learning rate range test.

        Args:
            skip_start (int, optional): Number of batches to trim from the start.
                (default: 10)
            skip_end (int, optional): Number of batches to trim from the end.
                (default: 5)
            log_lr (bool, optional): True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale. (default: True)
            show_lr (float, optional): Is set, will add vertical line to visualize
                specified learning rate. (default: None)
        """

        if show_lr is not None and not isinstance(show_lr, float):
            raise ValueError("show_lr must be float")

        # Get the data to plot from the history dictionary.
        lrs = self.history['lr']
        metrics = self.history['metric']

        # Plot metric_value as a function of the learning rate
        plt.plot(lrs, metrics)
        if log_lr:
            plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel(self.metric.title())

        if show_lr is not None:
            plt.axvline(x=show_lr, color='red')
        plt.show()


class LinearLR(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        end_lr (float): The final learning rate.
        iterations (int): The number of iterations over which the test occurs.
        last_epoch (int, optional): The index of last epoch. (default: -1)
    """

    def __init__(self, optimizer, end_lr, iterations, last_epoch=-1):
        self.end_lr = end_lr
        self.iterations = iterations
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.iterations
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        end_lr (float): The final learning rate.
        iterations (int): The number of iterations/epochs over which the test occurs.
        last_epoch (int, optional): The index of last epoch. (default: -1)
    """

    def __init__(self, optimizer, end_lr, iterations, last_epoch=-1):
        self.end_lr = end_lr
        self.iterations = iterations
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.iterations
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class StateCacher(object):
    def __init__(self, in_memory, cache_dir=None):
        self.in_memory = in_memory
        self.cache_dir = cache_dir

        if self.cache_dir is None:
            import tempfile

            self.cache_dir = tempfile.gettempdir()
        else:
            if not os.path.isdir(self.cache_dir):
                raise ValueError('Given cache_dir is not a valid directory.')

        self.cached = {}

    def store(self, key, state_dict):
        if self.in_memory:
            self.cached.update({key: copy.deepcopy(state_dict)})
        else:
            fn = os.path.join(self.cache_dir, f'state_{key}_{id(self)}.pt')
            self.cached.update({key: fn})
            torch.save(state_dict, fn)

    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError(f'Target {key} was not cached.')

        if self.in_memory:
            return self.cached.get(key)
        else:
            fn = self.cached.get(key)
            if not os.path.exists(fn):
                raise RuntimeError(
                    f"Failed to load state in {fn}. File doesn't exist anymore."
                )
            state_dict = torch.load(fn, map_location=lambda storage, location: storage)
            return state_dict

    def __del__(self):
        """Check whether there are unused cached files existing in cache_dir before
        this instance being destroyed.
        """

        if self.in_memory:
            return

        for k in self.cached:
            if os.path.exists(self.cached[k]):
                os.remove(self.cached[k])
