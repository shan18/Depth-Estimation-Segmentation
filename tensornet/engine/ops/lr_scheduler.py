from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR


def step_lr(optimizer, step_size, gamma=0.1, last_epoch=-1):
    """Create LR step scheduler.

    Args:
        optimizer (torch.optim): Model optimizer.
        step_size (int): Frequency for changing learning rate.
        gamma (float): Factor for changing learning rate. (default: 0.1)
        last_epoch (int): The index of last epoch. (default: -1)
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, factor=0.1, patience=10, verbose=False, min_lr=0):
    """Create LR plateau reduction scheduler.

    Args:
        optimizer (torch.optim): Model optimizer.
        factor (float, optional): Factor by which the learning rate will be reduced.
            (default: 0.1)
        patience (int, optional): Number of epoch with no improvement after which learning
            rate will be will be reduced. (default: 10)
        verbose (bool, optional): If True, prints a message to stdout for each update.
            (default: False)
        min_lr (float, optional): A scalar or a list of scalars. A lower bound on the
            learning rate of all param groups or each group respectively. (default: 0)
    
    Returns:
        ReduceLROnPlateau instance.
    """

    return ReduceLROnPlateau(
        optimizer, factor=factor, patience=patience, verbose=verbose, min_lr=min_lr
    )


def one_cycle_lr(
    optimizer, max_lr, epochs, steps_per_epoch, pct_start=0.5, div_factor=10.0, final_div_factor=10000
):
    """Create One Cycle Policy for Learning Rate.

    Args:
        optimizer (torch.optim): Model optimizer.
        max_lr (float): Upper learning rate boundary in the cycle.
        epochs (int): The number of epochs to train for. This is used along with
            steps_per_epoch in order to infer the total number of steps in the cycle.
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the cycle.
        pct_start (float, optional): The percentage of the cycle (in number of steps)
            spent increasing the learning rate. (default: 0.5)
        div_factor (float, optional): Determines the initial learning rate via
            initial_lr = max_lr / div_factor. (default: 10.0)
        final_div_factor (float, optional): Determines the minimum learning rate via
            min_lr = initial_lr / final_div_factor. (default: 1e4)
    
    Returns:
        OneCycleLR instance.
    """

    return OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
    )
