import torch
from torch import nn, Tensor
from typing import Tuple, Union, Callable, List
import numpy as np

import sys
sys.path.append('../../')

from utils import Timer

OptionalInt = Union[int, None]
OptionalStr = Union[str, None]
StrOrList = Union[str, List[str], Tuple[str], None]
OptimizerType = torch.optim.Optimizer
DataLoaderType = torch.utils.data.dataloader.DataLoader
LossFnType = Union[Callable[[nn.Module, Tensor], Tensor], Callable[[nn.Module, Tuple[Tensor, ...]], Tensor]]
BatchTensorType = Callable[[Tensor], Tuple[Tensor, ...]]

def train_sgd_auto(model: nn.Module, train_dataset: DataLoaderType, validate_dataset: DataLoaderType,
              test_dataset: DataLoaderType, loss_fn: LossFnType, quality_criterion: LossFnType, 
              batch_to_tensors: BatchTensorType, config_train: dict, save_path: OptionalStr = None, exp_name: OptionalStr = None, 
              save_every: OptionalInt = None, weight_names: StrOrList = None):
    """
    Function optimizes model parameters using common stochastic gradient descent, loss.backward() method.

    Args:
        model (nn.Module): The model with differentiable parameters.
        train_dataset (torch DataLoader type): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
        validate_dataset (torch DataLoader type, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate intermediate quality criterion values. 
            Attention! Validate dataset must have only 1 batch containing whole signal.
        test_dataset (DataLoader, optional): Batched dataset, prepared by the torch.utils.data.dataloader.DataLoader function.
            Current dataset is used to calculate quality criterion for test data.
            Attention! Test dataset must have only 1 batch containing whole signal, the same as for validation dataset.
        loss_fn (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar.
        quality_criterion (Callable): The function used to compute model quality. Takes nn.Module and tuple of two Tensor
            instances. Returns differentiable Tensor scalar. quality_criterion is not used in the model differentiation
            process, but it`s only used to estimate model quality in more reasonable units comparing to the loss_fn.
        batch_to_tensors (Callable): Function which acquires signal batch as an input and returns tuple of tensors, where
            the first tensor corresponds to model input, the second one - to the target signal.
        config_train (dictionary): Dictionary with configurations of training procedure. Includes learning rate, training type,
            optimizers parameters etc. Implied to be loaded from .yaml config file.
        save_path (str, optional): Folder path to save function product. Defaults to "None".
        exp_name (str, optional): Name of simulation, which is reflected in function product names. Defaults to "None".
        save_every (int, optional): The number which reflects following: the results would be saved every save_every epochs.
            If save_every equals None, then results will be saved at the end of learning. Defaults to "None".
        weight_names (str or list of str, optional): By spceifying `weight_names` it is possible to compute gradient only
            for several named parameters. Defaults to "None".

    Returns:
        Learning curve (list), containing quality criterion calculated each epoch of learning.
    """
    epochs = int(2e+4)

    if save_every is None:
        save_every = epochs - 1

    # Assign requires_grad of parameters, which are not within the weight_names to False
    for name, p in model.named_parameters():
        if name not in weight_names:
            p.requires_grad = False

    lrs = []
    learning_curve_train = []
    learning_curve_test = []
    learning_curve_validate = []
    learning_curve_train_qcrit = []
    learning_curve_test_qcrit = []
    learning_curve_validate_qcrit = []
    grad_norm_curve = []
    weights_norm_curve = []
    weight_decay = 0 # 1e-5
    # optimizer = torch.optim.SGD(model.parameters(), lr=5.e-0, momentum=0.99, weight_decay=weight_decay, nesterov=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-0, betas=(0.9, 0.9), weight_decay=weight_decay)
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-0, history_size=1000, max_iter=10, line_search_fn="strong_wolfe", tolerance_change=1e-40, tolerance_grad=1e-40)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, \
    #                                                        patience=epochs, threshold=1e-2, threshold_mode='abs')
    
    lambda_lin = lambda epoch: 1#1 - (1 - 1e-1)*epoch/epochs
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lin)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1e-5, total_iters=epochs)

    print_every = 1
    timer = Timer()
    general_timer = Timer()
    general_timer.__enter__()

    def accum_loss(dataset):
        loss_val = 0
        for batch in dataset:
            loss_val += loss_fn(model, batch).item()
        return loss_val

    # Calculate initial values of loss and quality criterion on validation and test dataset
    with torch.no_grad():
        loss_val_test = accum_loss(test_dataset)
        criterion_val_test = quality_criterion(model, test_dataset)
        best_criterion_test = criterion_val_test
        learning_curve_test.append(loss_val_test)
        learning_curve_test_qcrit.append(criterion_val_test)
        print("Begin: loss = {:.4e}, quality_criterion_test = {:.8f} dB.".format(loss_val_test, criterion_val_test))
        loss_val_train = accum_loss(train_dataset)
        criterion_val_train = quality_criterion(model, train_dataset)
        learning_curve_train.append(loss_val_train)   
        learning_curve_train_qcrit.append(criterion_val_train)
        print("Begin: loss = {:.4e}, quality_criterion_train = {:.8f} dB.".format(loss_val_train, criterion_val_train))
        loss_val_validate = accum_loss(validate_dataset)
        criterion_val_validate = quality_criterion(model, validate_dataset)
        learning_curve_validate.append(loss_val_validate)
        learning_curve_validate_qcrit.append(criterion_val_validate)
        print("Begin: loss = {:.4e}, quality_criterion_validate = {:.8f} dB.".format(loss_val_validate, criterion_val_validate))
    
    for epoch in range(epochs):
        timer.__enter__()
        for j, batch in enumerate(train_dataset):
            def closure():
                optimizer.zero_grad()
                loss_val = loss_fn(model, batch)
                loss_val.backward(create_graph=False)
                return loss_val
            optimizer.step(closure)
            scheduler.step()
            # scheduler.step(criterion_val)
        
        # Track NMSE values on validation and test dataset and save gradient, model parameters norm history
        with torch.no_grad():
            if epoch % save_every == 0:

                # Track algorithm parameters
                curr_params = torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad == True], dim=0)
                grad = torch.cat([p.grad.view(-1) for p in model.parameters() if p.requires_grad == True], dim=0)
                grad_distr = [torch.norm(p.grad).item() for p in model.parameters() if p.requires_grad == True]
                mu = scheduler.get_last_lr()[0]
                grad_norm = torch.norm(grad).item()
                grad_norm_curve.append(grad_norm)
                weights_norm_curve.append(torch.norm(curr_params).item())
                lrs.append(mu)

                loss_val_train = accum_loss(train_dataset)
                criterion_val_train = quality_criterion(model, train_dataset)
                loss_val_test = accum_loss(test_dataset)
                criterion_val_test = quality_criterion(model, test_dataset)
                loss_val_validate = accum_loss(validate_dataset)
                criterion_val_validate = quality_criterion(model, validate_dataset)

                learning_curve_test.append(loss_val_test)
                learning_curve_train.append(loss_val_train)
                learning_curve_validate.append(loss_val_validate)
                learning_curve_test_qcrit.append(criterion_val_test)
                learning_curve_train_qcrit.append(criterion_val_train)
                learning_curve_validate_qcrit.append(criterion_val_validate)

                if criterion_val_test < best_criterion_test:
                    best_criterion_test = criterion_val_test
                    torch.save(model.state_dict(), save_path+'weights_best_test'+exp_name)

                    np.save(save_path + f'lc_train{exp_name}.npy', np.array(learning_curve_train))
                    np.save(save_path + f'lc_test{exp_name}.npy', np.array(learning_curve_test))
                    np.save(save_path + f'lc_validate{exp_name}.npy', np.array(learning_curve_validate))
                    np.save(save_path + f'lc_qcrit_train{exp_name}.npy', np.array(learning_curve_train_qcrit))
                    np.save(save_path + f'lc_qcrit_test{exp_name}.npy', np.array(learning_curve_test_qcrit))
                    np.save(save_path + f'lc_qcrit_validate{exp_name}.npy', np.array(learning_curve_validate_qcrit))
                    np.save(save_path + f'grad_norm{exp_name}.npy', np.array(grad_norm_curve))
                    np.save(save_path + f'grad_distr{exp_name}.npy', np.array(grad_distr))
                    np.save(save_path + f'param_norm{exp_name}.npy', np.array(weights_norm_curve))
                    np.save(save_path + f'lrs{exp_name}.npy', np.array(lrs))
        timer.__exit__()
        if epoch % print_every == 0:
            print(f"Epoch is {epoch + 1}, " + \
                f"loss_train = {loss_val_train:.8f}, " + \
                f"quality_criterion_train = {criterion_val_train:.8f} dB, stepsize = {mu:.6e}, " + \
                f"|grad| = {grad_norm:.4e}, time elapsed: {timer.interval:.2e}")

        general_timer.__exit__()
        print(f"Total time elapsed: {general_timer.interval} s")
            
    general_timer.__exit__()
    print(f"Total time elapsed: {general_timer.interval} s")

    return learning_curve_test, best_criterion_test