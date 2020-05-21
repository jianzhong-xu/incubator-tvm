# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import onnx
import tvm
import tvm.relay.testing
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.contrib import util
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator
from tvm.contrib import cc
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime
import tvm.relay.op.contrib.tidl
from tvm.contrib.download import download_testdata

from tidl_tools import tidl
from tidl_tools import tidl_utils
from tidl_prune_subgraphs_example_v2 import PruneSubgraphs


class WholeGraphAnnotator(ExprMutator):
    """
    An annotator that creates a compiler for an entire graph.
    """

    def __init__(self, compiler):
        super(WholeGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call

# Leverage the pass manager to write a simple white list based annotator
@relay.transform.function_pass(opt_level=0)
class WhiteListAnnotator:
    def __init__(self, op_list, compiler):
        assert isinstance(op_list, (list, tuple, set))
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):

        annotator = self
        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_name = call.op.name
                if op_name in annotator.op_list:
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg),
                                             annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs,
                                          call.type_args)
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)
        return Annotator().visit(func)

def test_extern_tidl():
    target = "llvm -target=armv7l-linux-gnueabihf"

    #============= Constructing a simple graph ==============
    dtype = 'float32'
    data_layout = 'NCHW'
    input_shape    = (1, 3, 224, 224) # NCHW
    tidl_input_dim = (input_shape[2],input_shape[3],input_shape[1]) # HxWxC
    input_layout= 'NCHW'
    w1_shape    = (32, 3, 3, 3)    # OIHW
    w2_shape    = (1, 32, 3, 3)    # OIHW
    mnet1_conv2d_0 = np.load('MobilenetV1_Conv2d_0_weights.npy').astype(np.float32)
    # change layout from HWIO to OIHW
    w1 = mnet1_conv2d_0.transpose(3,2,0,1)
    params_w1 = tvm.nd.array(w1)
    mnet1_conv2d_1 = np.load('MobilenetV1_Conv2d_1_weights.npy').astype(np.float32) # HWIO
    w2 = mnet1_conv2d_1.transpose(3,2,0,1)
    params_w2 = tvm.nd.array(w2)

    data = relay.var('data', shape=(input_shape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)
    conv2d_1 = relay.nn.conv2d(data,
                               weight1,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               strides=(2,2),
                               data_layout = data_layout,
                               kernel_layout = 'OIHW')
    weight2 = relay.var('weight2', shape=(w2_shape), dtype=dtype)
    conv2d_2 = relay.nn.conv2d(conv2d_1,
                               weight2,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               data_layout = data_layout,
                               kernel_layout = 'OIHW')
    clip = relay.clip(conv2d_2,0,6)
    #out = conv2d_1
    out = conv2d_2
    #out  = clip
    f1 = relay.Function([data, weight1, weight2], out)
    mod1 = tvm.IRModule.from_expr(f1)
    params0 = {'weight1':params_w1, 'weight2':params_w2} 
    print('---------- Original graph ----------')
    print(mod1.astext(show_meta_data=False))

    #============= Build the graph to run on ARM =============
    print('Build the graph to run on ARM')
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build_module.build(mod1, target=target, params=params0)

    artifacts_folder = "./artifacts_arm/"
    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    lib.export_library(path_lib, cc=arm_gcc)
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))
    print('Model artifacts saved to run on ARM')

    #============= Annotating the graph to run on TIDL ==============
    mod1['main'] = bind_params_by_name(mod1['main'], params0)
    # whole graph offload to TIDL
    mod2 = tvm.IRModule()
    mod2['main'] = WholeGraphAnnotator('tidl').visit(mod1['main'])
    print('---------- Whole graph annotated ----------')
    print(mod2.astext(show_meta_data=False))
    mod_whole_graph_tidl = relay.transform.PartitionGraph()(mod2)
    print('---------- Whole graph annotated and partitioned ----------')
    print(mod_whole_graph_tidl.astext(show_meta_data=False))

    # subraph offload to TIDL
    mod3 = WhiteListAnnotator(["nn.conv2d"],"tidl")(mod1)
    print('---------- Subgraph annotated ----------')
    print(mod3.astext(show_meta_data=False))
    mod_subgraph_tidl = relay.transform.PartitionGraph()(mod3)
    print('---------- Subgraph annotated and partitioned ----------')
    print(mod_subgraph_tidl.astext(show_meta_data=False))

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    #mod4 = relay.transform.InferType()(mod_whole_graph_tidl)
    mod4 = relay.transform.InferType()(mod_subgraph_tidl)
    mod4 = relay.transform.Inline()(mod4)
    my_mutator = tidl.CalibrationGraphMutator("tidl")
    mod4["main"] = my_mutator.make_calibration_graph(mod4["main"])
    print("Calibration module:", mod4)
    print("Input map:", my_mutator.name_map)

    # Build and execute calibration graph to get outputs
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod4, "llvm", params=params0)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    x = np.load('./tidl_tools/dog.npy') # (1,3,224,224)
    input_data = x/np.amax(np.abs(x))
    print("Input data shape: ")
    print(input_data.shape)
    mod.set_input('data', input_data)
    mod.set_input(**params)
    mod.run()
    #mod.run(data=input_data, weight1=params_w1, weight2=params_w2)

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    np.savetxt('graph_output.txt', results[0].flatten(), fmt='%10.5f')

    # We now have subgraph inputs
    # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
    subgraph_tensors = {}
    for i in range(len(results)):
        if i in my_mutator.name_map:
            subgraph_tensors[my_mutator.name_map[i]]=results[i]
            #print("Subgraph input: ", my_mutator.name_map[i], " tensor: ", results[i])
            file_name = my_mutator.name_map[i] + ".txt"
            np.savetxt(file_name, results[i].flatten(), fmt='%10.5f')

    for key, value in subgraph_tensors.items():
        print("Subgraph tensor: ", key, value.shape)

    #======================== Import the graph to TIDL ========================
    mod2 = mod_subgraph_tidl
    #mod2 = mod_whole_graph_tidl
    if tidl.import_relay_ir(mod2, params0, subgraph_tensors, data_layout, tidl_calib_tool, artifacts_folder) == True:
        print('Heterogeneous execution with TIDL.')
        graph, lib, params = relay.build_module.build(mod2, target=target, params=params0)
    else:
        print("Full graph compilation with LLVM.")
        # Future optimization: if not all subgraphs failed with TIDL import, re-partition
        # the graph to have only TIDL subgraphs with successful TIDL import. 
        graph, lib, params = relay.build_module.build(mod1, target=target, params=params0)

    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    #lib.save(path_lib) # for whole graph execute on TIDL
    lib.export_library(path_lib, cc=arm_gcc) # for heterogeneous execute on TIDL+ARM
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


import tensorflow as tf
from tvm.relay.testing import tf as tf_testing

def create_tf_relay_graph(model, input_node, input_shape, layout):

    if model == "MobileNetV1":
        model    = "./mobileNet1/mobilenet_v1_1.0_224_frozen.pb"
        out_node = 'MobilenetV1/Predictions/Softmax'
    elif model == "MobileNetV2":
        model    = "./mobileNet2/mobilenet_v2_1.0_224_frozen.pb"
        out_node = 'MobilenetV2/Predictions/Softmax'
    elif model == "InceptionV1":
        model    = "./inception1/inception_v1_fbn.pb"
        out_node = "softmax/Softmax"
    elif model == "InceptionV3":
        model    = "./inception3/inception_v3_2016_08_28_frozen-with_shapes.pb"
        out_node = "InceptionV3/Predictions/Softmax"
    #if layout == "NCHW":
    #    input_shape = (input_shape[0],input_shape[2],input_shape[3],input_shape[1])
    with tf.gfile.GFile(model, 'rb') as f:
        # Import tensorflow graph definition to relay frontend.
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        # Add shapes to the graph.
        with tf.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)

        shape_dict = {input_node : input_shape}
        print("Inut node shape dict:" + str(shape_dict))
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                     layout = None,  # default: NHWC
                                                     shape  = shape_dict, 
                                                     outputs= None)
        mod = relay.transform.RemoveUnusedFunctions()(mod)
        print("Tensorflow model imported to Relay IR.")

    return mod, params


def generate_subgraph_tensors(mod, params, input_node, input_data):
    """
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    my_mutator = tidl.CalibrationGraphMutator("tidl")
    mod_tvm["main"] = my_mutator.make_calibration_graph(mod_tvm["main"])
    #print("Calibration module:", mod_tvm)
    print("Input map:", my_mutator.name_map)

    # Build and execute calibration graph to get outputs
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(input_node, input_data)
    mod.set_input(**params)
    mod.run()
    #mod.run(data=input_data, weight1=params_w1, weight2=params_w2)

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    np.savetxt('graph_output.txt', results[0].flatten(), fmt='%10.5f')

    # We now have subgraph inputs
    # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
    subgraph_tensors = {}
    for i in range(len(results)):
        if i in my_mutator.name_map:
            subgraph_tensors[my_mutator.name_map[i]]=results[i]
            #print("Subgraph input: ", my_mutator.name_map[i], " tensor: ", results[i])
            file_name = my_mutator.name_map[i] + ".txt"
            np.savetxt(file_name, results[i].flatten(), fmt='%10.5f')

    for key, value in subgraph_tensors.items():
        print("Subgraph tensor: ", key, value.shape)

    return subgraph_tensors

def tidl_codegen(mod_orig, params_orig, num_tidl_subgraphs, data_layout, input_node, input_data):

    mod_orig = relay.transform.RemoveUnusedFunctions()(mod_orig)
    # Bind params so that weights will appear as constants instead of variables 
    mod_orig['main'] = bind_params_by_name(mod_orig['main'], params_orig)
    print("---------- Original graph ----------")
    print(mod_orig.astext(show_meta_data=False))

    #============= Annotate the graph ==============
    # Looks at annotated ops and marks them in the graph with compiler.begin 
    # and compiler.end.
    # Merges annotated regions together that use the same external target, 
    # and combines marked regions for each target
    #Merge sequence of ops into composite functions/ops
    print("---------- Merge Composite Functions ----------")
    mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod_orig) 
    print("---------- Annotated Graph ----------")
    mod = transform.AnnotateTarget("tidl")(mod)
    print("---------- Merge Compiler Regions ----------")
    mod = transform.MergeCompilerRegions()(mod)
    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    print("---------- Unpack composite ops in the graph ----------")
    mod = tidl.UnpackComposites(mod, "tidl")
    print("---------- Prune Graph ----------")
    mod = PruneSubgraphs(mod, compiler="tidl", num_subgraphs_to_keep=num_tidl_subgraphs)
    print(mod.astext(show_meta_data=False))

    #============= Generate subgraph boundary tensors ==============
    subgraph_tensors = generate_subgraph_tensors(mod, params_orig, input_node, input_data)

    #======================== Import the graph to TIDL ========================
    if tidl.import_relay_ir(mod, params_orig, subgraph_tensors, data_layout, tidl_calib_tool, artifacts_folder) == True:
        print("Graph execution with TIDL.")
    else:
        print("Graph execution with general CPU.")
        mod = mod_orig

    graph, lib, params = relay.build_module.build(mod, target=target, params=params_orig)
    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    #lib.save(path_lib) # for whole graph execute on TIDL
    lib.export_library(path_lib, cc=arm_gcc) # for heterogeneous execute on TIDL+ARM
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


def test_extern_tidl_tf(tf_model, num_tidl_subgraphs):
    dtype = "float32"
    data_layout = "NHWC"
    input_shape = (1, 224, 224, 3)
    x = np.load('./tidl_tools/dog.npy')  # "NCHW"
    x = x.transpose(0,2,3,1)  # TF uses "NHWC" layout
    if x.shape != input_shape:
        sys.exit("Input data shape is not correct!")
    # Normalize input data to (-1,1)
    input_data = x/np.amax(np.abs(x))
    input_node = "input"

    #============= Create a Relay graph for MobileNet model ==============
    tf_mod, tf_params = create_tf_relay_graph(model = tf_model,
                                              input_node  = input_node,
                                              input_shape = input_shape,
                                              layout = data_layout)
    print("---------- Original TF Graph ----------")
    print(tf_mod.astext(show_meta_data=False))

    #======================== TIDL code generation ====================
    tidl_codegen(tf_mod, tf_params, num_tidl_subgraphs, data_layout, input_node, input_data)

def test_extern_tidl_onnx(onnx_model, num_tidl_subgraphs):
    model_path = "./onnx_resNet18v2/resnet18v2.onnx"
    #model_path = "./onnx_squeezeNet/squeezenet1.1.onnx"
    onnx_model = onnx.load(model_path)

    image_shape = (1, 3, 224, 224)
    data_layout = "NCHW"
    x = np.load('./tidl_tools/dog.npy')  # "NCHW"
    input_data = x/np.amax(np.abs(x))

    input_name = "data"
    shape_dict = {input_name: image_shape }
    onnx_mod, onnx_params = relay.frontend.from_onnx(onnx_model, shape_dict)

    #======================== TIDL code generation ====================
    tidl_codegen(onnx_mod, onnx_params, num_tidl_subgraphs, data_layout, input_name, input_data)

from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.presets.segmentation import test_transform
import mxnet as mx
from mxnet import image

def load_gluoncv_model(model, x, input_name, input_shape, dtype):
    block = model_zoo.get_model(model, pretrained=True)

    if 'faster' in model:
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
    else:
        block.hybridize()
        block.forward(x)
        block.export('temp') # create file temp-symbol.json and temp-0000.params

        model_json = mx.symbol.load('temp-symbol.json')
        save_dict = mx.ndarray.load('temp-0000.params')
        arg_params = {}
        aux_params = {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            elif tp == 'aux':
                aux_params[name] = v
        mod, params = relay.frontend.from_mxnet(model_json, {input_name: input_shape}, arg_params=arg_params, aux_params=aux_params)
    return block, mod, params

def test_extern_tidl_gluoncv_ssd(model, num_tidl_subgraphs):
    im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                                 'gluoncv/detection/street_small.jpg?raw=true',
                                 'street_small.jpg', module='data')
    image_size = 512
    data_layout = "NCHW"
    dtype = "float32"
    input_name = "data"

    #======================== Load testing image and model ====================
    input_shape = (1, 3, image_size, image_size)
    x, img = data.transforms.presets.ssd.load_test(im_fname, short=image_size)
#    block = model_zoo.get_model(model, pretrained=True)
#    block.hybridize()
#    block.forward(x)
#    block.export('temp')
#    ssd_mod, ssd_params = relay.frontend.from_mxnet(block, {"data": input_shape})
    block, ssd_mod, ssd_params = load_gluoncv_model(model, x, input_name, input_shape, dtype)

    #======================== Execute the full graph on TVM ====================
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(ssd_mod, "llvm", params=ssd_params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    input_data = x.asnumpy()
    np.save("ssd_input.npy",input_data) # to be used by inference testing on the target
    mod.set_input(input_name, input_data)
    mod.set_input(**params)
    mod.run()
    class_IDs, scores, bounding_boxs = mod.get_output(0), mod.get_output(1), mod.get_output(2)
    ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                             class_IDs.asnumpy()[0], class_names=block.classes)
    plt.savefig("gluoncv_ssd_tvm.png")
    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    print("Number of outputs: " + str(len(results)))
    #for i in range(len(results)):
    #    np.savetxt("graph_out_"+str(i)+".txt", results[i].flatten(), fmt='%10.5f')

    #======================== TIDL code generation ====================
    tidl_codegen(ssd_mod, ssd_params, num_tidl_subgraphs, data_layout, input_name, input_data)

def test_extern_tidl_gluoncv_segmentation(model, num_tidl_subgraphs):
    input_name = "data"
    data_layout = "NCHW"
    dtype = "float32"

    #======================== Load testing image and model ====================
    img_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'
    filename = 'example.jpg'
    utils.download(img_url, filename)
    img = image.imread(filename)
    img = test_transform(img, ctx = mx.cpu(0))
    input_shape = img.shape
    input_data = img.asnumpy()
    np.save("seg_input.npy",input_data) # to be used by inference testing on the target
    model = model_zoo.get_model(model, pretrained=True)
    model.hybridize()
    model.forward(img)
    model.export('gluoncv-temp')    # create file gluoncv-temp-symbol.json
    seg_mod, seg_params = relay.frontend.from_mxnet(model, {input_name:input_shape})
#    block, seg_mod, seg_params = load_gluoncv_model(model, img, input_name, input_shape, dtype)

    #======================== TIDL code generation ====================
    tidl_codegen(seg_mod, seg_params, num_tidl_subgraphs, data_layout, input_name, input_data)

if __name__ == '__main__':
    if os.getenv("TIDL_ARM_GCC_PATH") is None:
      sys.exit("Environment variable TIDL_ARM_GCC_PATH not set!")
    else: 
      arm_gcc_path = os.getenv("TIDL_ARM_GCC_PATH")
    if os.getenv("TIDL_TOOLS_PATH") is None:
        sys.exit("Environment variable TIDL_TOOLS_PATH not set!")
    else:
        tidl_tools_path = os.getenv("TIDL_TOOLS_PATH")
    arm_gcc          = os.path.join(arm_gcc_path, "arm-linux-gnueabihf-g++")
    tidl_calib_tool  = os.path.join(tidl_tools_path, "eve_test_dl_algo_ref.out")
    target = "llvm -target=armv7l-linux-gnueabihf"

    artifacts_folder = "./artifacts/"
    if os.path.isdir(artifacts_folder):
        filelist = [ f for f in os.listdir(artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(artifacts_folder, file))
    else:
        os.mkdir(artifacts_folder)

    ssd_models = ['ssd_512_mobilenet1.0_coco',
                  'ssd_512_mobilenet1.0_voc',
                 ]
    seg_models = ['mask_rcnn_resnet18_v1b_coco',
                 ]

    #test_extern_tidl()
    #test_extern_tidl_tf("MobileNetV2", num_tidl_subgraphs=1)
    #test_extern_tidl_onnx("resnet18", num_tidl_subgraphs=1)
    test_extern_tidl_gluoncv_ssd(ssd_models[1], num_tidl_subgraphs=1)
    #test_extern_tidl_gluoncv_segmentation(seg_models[0], num_tidl_subgraphs=1)
