import os

import numpy as np

import tvm
from tvm import relay
from tvm import autotvm
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import torch
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime

from craft import CRAFT

from collections import OrderedDict

#### DEVICE CONFIG ####
target = tvm.target.create("llvm -target=armv7l-linux-gnueabihf") # tvm.target.arm_cpu('rasp3b')
device_key = 'sitara'
dtype = 'float32'
cc = 'arm-linux-gnueabihf-g++'

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
#################################################################
# Define Network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`nnvm.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

def assert_shapes_match(tru, est):
    print(tru.shape)
    print(est.shape)
    print(tru.shape == est.shape)
    if tru.shape != est.shape:
        msg = "Output shapes {} and {} don't match"
        raise AssertionError(msg.format(tru.shape, est.shape))

def get_network(batch_size):
    """Get the symbol definition and random weight of a network"""
    with torch.no_grad():
        filename = 'craft_mlt_25k.pth'
        input_shape = (batch_size, 3, 768, 768)
        output_shape = (batch_size, 1000)
        print(input_shape)
        input_data = torch.randn(input_shape)
        net = CRAFT()
        print('Loading weights from checkpoint')
        net.load_state_dict(copyStateDict(torch.load(filename, map_location='cpu')))
        net.eval()
        trace = torch.jit.trace(net, input_data).eval()
        #trace = torch.jit.load('saved_craft_mlt_25k.pth')
        #torch.jit.save(trace, 'saved_craft_mlt_25k.pth')
        input_name = "input0"
        input_names = [input_name]
        baseline_input = [input_data]
        print('Import graph to relay')
        mod, params = relay.frontend.from_pytorch(trace, [(input_name, input_shape)])

        baseline_outputs = trace.forward(input_data)
        print(baseline_outputs)
        baseline_outputs = tuple(out.cpu().numpy() for out in baseline_outputs)
        compiled_input = dict(zip(input_names,
                                  [inp.cpu().numpy() for inp in baseline_input]))

        #Check accuracy
        with relay.build_config(opt_level=3):
            for target, ctx in ctx_list():
                relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)
                relay_model = graph_runtime.create(relay_graph, relay_lib, ctx)
                relay_model.set_input(**relay_params)
                for name, inp in compiled_input.items():
                    relay_model.set_input(name, inp)
                relay_model.run()

                for i, baseline_output in enumerate(baseline_outputs):
                    compiled_output = relay_model.get_output(i).asnumpy()

                    assert_shapes_match(baseline_output, compiled_output)
                    tvm.testing.assert_allclose(baseline_output, compiled_output,
                                                rtol=1e-3, atol=1e-3)
                print('No accuracy issues')

        return mod, params

def compile():
    # extract workloads from relay program
    mod, params = get_network(batch_size=1)

    print("Compile...")
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)

    # export library
    lib.export_library("deploy_lib.so", cc=cc)
    with open("deploy_graph.json", "w") as fo:
        fo.write(graph)
    with open("deploy_param.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))

#TODO: Remove this file before PR
compile()
