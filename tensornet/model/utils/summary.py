# The code in this file has been referenced from
# https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py


import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, dtypes=None):
    """Display model summary.

    Args:
        model (torch.nn.Module): Model instance.
        input_size (tuple, list or dict): Input size for the model.
        batch_size (int, optional): Batch size. (default: -1)
        dtypes (optional): Model input data types. (default: None)
    """
    device = next(model.parameters()).device
    result, _ = summary_string(
        model, input_size, device, batch_size=batch_size, dtypes=dtypes
    )
    print(result)


def summary_string(model, input_size, device, batch_size=-1, dtypes=None):
    """Prepare model summary.

    Args:
        model (torch.nn.Module): Model instance.
        input_size (tuple, list or dict): Input size for the model.
        device (torch.device, optional): Device.
        batch_size (int, optional): Batch size. (default: -1)
        dtypes (optional): Model input data types. (default: None)
    
    Returns:
        Model summary and number of parameters in the model
    """
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            if isinstance(input[0], dict):
                summary[m_key]['input_shape'] = [
                    [batch_size] + list(input[0][key].size())[1:] for key in input[0]
                ]
            else:
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, dict):
                summary[m_key]['output_shape'] = [
                    [-1] + list(output[key].size())[1:] for key in
                    output.keys()
                ]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = batch_size

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    if isinstance(input_size, dict):  # Inputs to the model are passed as a dict
        x = {
            in_size: torch.rand(
                2, *input_size[in_size]).type(dtype).to(device=device)
            for in_size, dtype in zip(input_size, dtypes)
        }
    else:  # Inputs to the model are passed as a list
        x = [
            torch.rand(2, *in_size).type(dtype).to(device=device)
            for in_size, dtype in zip(input_size, dtypes)
        ]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    if isinstance(x, dict):
        model(x)
    else:
        model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += '----------------------------------------------------------------' + '\n'
    line_new = f'{"Layer (type)":>20}  {"Output Shape":>25} {"Param #":>15}'
    summary_str += line_new + '\n'
    summary_str += '================================================================' + '\n'
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        output_shape = summary[layer]['output_shape']
        nb_params = summary[layer]['nb_params']
        line_new = f'{layer:>20}  {str(output_shape):>25} {f"{nb_params:,}":>15}'
        total_params += nb_params

        total_output += np.prod(output_shape)
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += nb_params
        summary_str += line_new + '\n'

    # assume 4 bytes/number (float on cuda).
    if isinstance(input_size, dict):
        total_input_size = abs(
            np.prod(sum(
                [input_size[key] for key in input_size], ()
            )) * batch_size * 4. / (1024 ** 2.)
        )
    else:
        total_input_size = abs(
            np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.)
        )
    total_output_size = abs(
        2. * total_output * 4. / (1024 ** 2.)
    )  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += '================================================================' + '\n'
    summary_str += f'Total params: {total_params:,}\n'
    summary_str += f'Trainable params: {trainable_params:,}\n'
    summary_str += f'Non-trainable params: {total_params - trainable_params:,}\n'
    summary_str += '----------------------------------------------------------------' + '\n'
    summary_str += 'Input size (MB): %0.2f' % total_input_size + '\n'
    summary_str += 'Forward/backward pass size (MB): %0.2f' % total_output_size + '\n'
    summary_str += 'Params size (MB): %0.2f' % total_params_size + '\n'
    summary_str += 'Estimated Total Size (MB): %0.2f' % total_size + '\n'
    summary_str += '----------------------------------------------------------------' + '\n'

    # return summary
    return summary_str, (total_params, trainable_params)
