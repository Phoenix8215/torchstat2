
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from torchstat2 import compute_madd
from torchstat2 import compute_flops
from torchstat2 import compute_memory


class ModelHook(object):
    def __init__(self, model, input_size):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (list, tuple))

        self._model = model
        self._input_size = input_size
        self._hooks = []  # Store registered forward hooks instead of modifying class methods

        self._hook_model()
        x = torch.rand(1, *self._input_size)  # add module duration time
        self._model.eval()
        
        # Try forward pass, and if it fails, try to adapt the input
        try:
            self._model(x)
        except RuntimeError as e:
            if "shapes cannot be multiplied" in str(e) or "size mismatch" in str(e):
                # Try flattening the input (common for MLP models)
                try:
                    flat_size = 1
                    for dim in self._input_size:
                        flat_size *= dim
                    x_flat = torch.rand(1, flat_size)
                    self._model(x_flat)
                    # If successful, update the input size
                    self._input_size = (flat_size,)
                    x = x_flat
                except:
                    # If still fails, re-raise the original error
                    raise e
            else:
                raise e

    @staticmethod
    def _register_buffer(module):
        assert isinstance(module, nn.Module)

        if len(list(module.children())) > 0:
            return

        # Check if buffers already exist before registering to avoid "already exists" error
        buffers_to_register = [
            ('input_shape', torch.zeros(3).int()),
            ('output_shape', torch.zeros(3).int()),
            ('parameter_quantity', torch.zeros(1).int()),
            ('inference_memory', torch.zeros(1).long()),
            ('MAdd', torch.zeros(1).long()),
            ('duration', torch.zeros(1).float()),
            ('Flops', torch.zeros(1).long()),
            ('Memory', torch.zeros(2).long())
        ]
        
        for buffer_name, buffer_tensor in buffers_to_register:
            if not hasattr(module, buffer_name):
                module.register_buffer(buffer_name, buffer_tensor)

    def _create_forward_hook(self, module):
        """Create forward pre and post hooks for a specific module instance"""

        def forward_pre_hook(module, input):
            """Record start time before module forward execution"""
            module._start_time = time.perf_counter()
        def forward_hook(module, input, output):
            """Collect statistics and record duration after forward execution"""
            # Find the first tensor input for getting itemsize and shape
            first_tensor_input = None
            for inp in input:
                if isinstance(inp, torch.Tensor):
                    first_tensor_input = inp
                    break
            
            # Default itemsize if no tensor found
            itemsize = first_tensor_input.element_size() if first_tensor_input is not None else 4

            end = time.perf_counter()
            start = getattr(module, "_start_time", end)
            module.duration = torch.from_numpy(
                np.array([end - start], dtype=np.float32))

            # Use first tensor input for shape if available, otherwise use default shape
            if first_tensor_input is not None:
                module.input_shape = torch.from_numpy(
                    np.array(first_tensor_input.size()[1:], dtype=np.int32))
            else:
                # Default shape for non-tensor inputs
                module.input_shape = torch.from_numpy(np.array([1], dtype=np.int32))
                
            # Handle output shape - check if output is a tensor
            if isinstance(output, torch.Tensor):
                module.output_shape = torch.from_numpy(
                    np.array(output.size()[1:], dtype=np.int32))
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                module.output_shape = torch.from_numpy(
                    np.array(output[0].size()[1:], dtype=np.int32))
            else:
                # Default shape for non-tensor outputs
                module.output_shape = torch.from_numpy(np.array([1], dtype=np.int32))

            parameter_quantity = 0
            # iterate through parameters and count num params
            for name, p in module._parameters.items():
                parameter_quantity += (0 if p is None else torch.numel(p.data))
            module.parameter_quantity = torch.from_numpy(
                np.array([parameter_quantity], dtype=np.int64))

            # Calculate inference memory more accurately
            # Include both output tensor size and parameter memory
            if isinstance(output, torch.Tensor):
                output_elements = output.numel()  # Total elements including batch dimension
                output_bytes = output_elements * output.element_size()  # Actual memory size
            elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                output_elements = output[0].numel()
                output_bytes = output_elements * output[0].element_size()
            else:
                # Default for non-tensor outputs
                output_elements = 1
                output_bytes = 4  # Assume 4 bytes per element
            
            # Add parameter memory for this module
            param_bytes = 0
            for param in module.parameters():
                if param is not None:
                    param_bytes += param.numel() * param.element_size()
            
            # Total inference memory in MB (output activations + parameters)
            total_bytes = output_bytes + param_bytes
            inference_memory = total_bytes / (1024 ** 2)  # Convert to MB
            
            module.inference_memory = torch.from_numpy(
                np.array([inference_memory], dtype=np.float32))

            # Compute statistics only if we have valid tensor inputs and outputs
            if first_tensor_input is not None and isinstance(output, (torch.Tensor, tuple, list)):
                try:
                    if len(input) == 1:
                        madd = compute_madd(module, input[0], output)
                        flops = compute_flops(module, input[0], output)
                        Memory = compute_memory(module, input[0], output)
                    elif len(input) > 1:
                        madd = compute_madd(module, input, output)
                        flops = compute_flops(module, input, output)
                        Memory = compute_memory(module, input, output)
                    else:  # error
                        madd = 0
                        flops = 0
                        Memory = (0, 0)
                except Exception as e:
                    # If computation fails, use default values
                    print(f"Warning: Failed to compute statistics for {module.__class__.__name__}: {e}")
                    madd = 0
                    flops = 0
                    Memory = (0, 0)
            else:
                # No valid tensor inputs/outputs
                madd = 0
                flops = 0
                Memory = (0, 0)
            module.MAdd = torch.from_numpy(
                np.array([madd], dtype=np.int64))
            module.Flops = torch.from_numpy(
                np.array([flops], dtype=np.int64))
            Memory = np.array(Memory, dtype=np.int32) * itemsize
            module.Memory = torch.from_numpy(Memory)

        return forward_pre_hook, forward_hook

    def _register_forward_hooks(self):
        """Register forward pre and post hooks on all leaf modules"""
        for module in self._model.modules():
            if len(list(module.children())) == 0:  # leaf module
                pre_hook, post_hook = self._create_forward_hook(module)
                h1 = module.register_forward_pre_hook(pre_hook)
                h2 = module.register_forward_hook(post_hook)
                self._hooks.extend([h1, h2])
    
    def __del__(self):
        """Remove all registered hooks when the object is destroyed"""
        self.remove_hooks()

    def remove_hooks(self):
        """Remove all registered forward hooks"""
        try:
            for hook in self._hooks:
                hook.remove()
            self._hooks.clear()
        except:
            # Ignore errors during cleanup
            pass

    def _hook_model(self):
        self._model.apply(self._register_buffer)
        self._register_forward_hooks()

    @staticmethod
    def _retrieve_leaf_modules(model):
        leaf_modules = []
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                leaf_modules.append((name, m))
        return leaf_modules

    def retrieve_leaf_modules(self):
        return OrderedDict(self._retrieve_leaf_modules(self._model))
