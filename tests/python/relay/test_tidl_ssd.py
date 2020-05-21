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

import gluoncv
import os
import sys
import numpy as np
import subprocess
from matplotlib import pyplot as plt

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.relay import transform
import tvm.relay.op.contrib.tidl
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib import graph_runtime

from tidl_tools import tidl

# For darknet tests
#Using darknet...
import sys
from tvm.contrib.download import download_testdata
download_testdata.__test__ = False
from tvm.relay.testing.darknet import LAYERTYPE
from tvm.relay.testing.darknet import __darknetffi__
from tvm.relay.frontend.darknet import ACTIVATION
import mxnet as mx
from mxnet import image
from gluoncv import model_zoo, data, utils
from tidl_prune_subgraphs_example_v2 import PruneSubgraphs
from gluoncv.data.transforms.presets.segmentation import test_transform

# Darknet
REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'

DARKNET_LIB = 'libdarknet_mac2.0.so'
DARKNETLIB_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'

if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
    DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

#LIB = __darknetffi__.dlopen(download_testdata(DARKNETLIB_URL, DARKNET_LIB, module='darknet'))
#
#DARKNET_TEST_IMAGE_NAME = 'dog.jpg'
#DARKNET_TEST_IMAGE_URL = REPO_URL + 'data/' + DARKNET_TEST_IMAGE_NAME +'?raw=true'
#DARKNET_TEST_IMAGE_PATH = download_testdata(DARKNET_TEST_IMAGE_URL, DARKNET_TEST_IMAGE_NAME, module='data')

# MxNet
im_fname = download_testdata('https://github.com/dmlc/web-data/blob/master/' +
                             'gluoncv/detection/street_small.jpg?raw=true',
                             'street_small.jpg', module='data')


def test_tidl_annotation():

    dtype = "float32"
    input_shape = (1, 3, 224, 224) # NCHW
    w1_shape    = (32, 3, 3, 3)    # OIHW

    data = relay.var('data', shape=(input_shape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1_shape), dtype=dtype)
    conv2d_1 = relay.nn.conv2d(data,
                               weight1,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               kernel_layout = 'OIHW')

    clip_1 = relay.clip(conv2d_1, 0, 6) # relu6

    squeeze_1 = relay.squeeze(clip_1)
    reshape_1 = relay.reshape(squeeze_1, (3, 3, 3, 32))

    out = reshape_1
    f1 = relay.Function([data, weight1], out)
    mod = tvm.IRModule.from_expr(f1)
    print('---------- Original graph ----------')
    print(mod.astext(show_meta_data=False))

    print("----------  Graph with composite fns ----------")
    #TODO: Uncomment after Cody refactor PR in
    mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod)
    print(mod.astext(show_meta_data=False))

    print("---------- Annotated graph ----------")
    mod = transform.AnnotateTarget("tidl")(mod)
    print(mod.astext(show_meta_data=False))

    print("---------- Annotated graph after merging ----------")
    mod = transform.MergeCompilerRegions()(mod)
    print(mod.astext(show_meta_data=False))

    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    print(mod.astext(show_meta_data=False))

def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

def test_tidl_mobilenet():
    import tensorflow as tf
    import tvm.relay.testing.tf as tf_testing

    with tf.Graph().as_default():

        graph_def = tf_testing.get_workload(
            "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
            "mobilenet_v2_1.4_224_frozen.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
        out_node = 'MobilenetV2/Predictions/Reshape_1'
        with tf.Session() as sess:
            # Add shapes to the graph.
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
            input_data = convert_to_list(data)
            input_node = convert_to_list('input')
            shape_dict = {e: i.shape for e, i in zip(input_node, input_data)}
            mod1, params = relay.frontend.from_tensorflow(graph_def,
                                                          shape=shape_dict)
            print('---------- Original Graph ----------')
            mod1 = relay.transform.RemoveUnusedFunctions()(mod1)
            print(mod1.astext(show_meta_data=False))
            print('---------- Merge Composite Functions ----------')
            mod3 = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod1) #Merge sequence of ops into composite functions/ops
            print(mod3.astext(show_meta_data=False))
            print("---------- Annotated Graph ----------")
            mod4 = transform.AnnotateTarget("tidl")(mod3) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
            print(mod4.astext(show_meta_data=False))
            print("---------- Merge Compiler Regions ----------")
            mod4 = transform.MergeCompilerRegions()(mod4) #Merge annotated regions together that use the same external target, combines marked regions for each target
            print(mod4.astext(show_meta_data=False))
            print("---------- Partioned Graph ----------")
            mod4 = transform.PartitionGraph()(mod4)
            print(mod4.astext(show_meta_data=False))

"""
def _read_memory_buffer(shape, data, dtype='float32'):
    length = 1
    for x in shape:
        length *= x
    data_np = np.zeros(length, dtype=dtype)
    for i in range(length):
        data_np[i] = data[i]
    return data_np.reshape(shape)

def _load_net(cfg_url, cfg_name, weights_url, weights_name):
    cfg_path = download_testdata(cfg_url, cfg_name, module='darknet')
    weights_path = download_testdata(weights_url, weights_name, module='darknet')
    net = LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)
    return net

def verify_darknet_frontend(net, build_dtype='float32'):
    '''Test network with given input image on both darknet and tvm'''
    def get_darknet_output(net, img):
        LIB.network_predict_image(net, img)
        out = []
        for i in range(net.n):
            layer = net.layers[i]
            if layer.type == LAYERTYPE.REGION:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.coords, layer.background],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.n*2, ), layer.biases))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif layer.type == LAYERTYPE.YOLO:
                attributes = np.array([layer.n, layer.out_c, layer.out_h,
                                       layer.out_w, layer.classes,
                                       layer.total],
                                      dtype=np.int32)
                out.insert(0, attributes)
                out.insert(0, _read_memory_buffer((layer.total*2, ), layer.biases))
                out.insert(0, _read_memory_buffer((layer.n, ), layer.mask, dtype='int32'))
                layer_outshape = (layer.batch, layer.out_c,
                                  layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(layer_outshape, layer.output))
            elif i == net.n-1:
                if layer.type == LAYERTYPE.CONNECTED:
                    darknet_outshape = (layer.batch, layer.out_c)
                elif layer.type in [LAYERTYPE.SOFTMAX]:
                    darknet_outshape = (layer.batch, layer.outputs)
                else:
                    darknet_outshape = (layer.batch, layer.out_c,
                                        layer.out_h, layer.out_w)
                out.insert(0, _read_memory_buffer(darknet_outshape, layer.output))
        return out

    dtype = 'float32'

    img = LIB.letterbox_image(LIB.load_image_color(DARKNET_TEST_IMAGE_PATH.encode('utf-8'), 0, 0), net.w, net.h)
    darknet_output = get_darknet_output(net, img)
    batch_size = 1
    data = np.empty([batch_size, img.c, img.h, img.w], dtype)
    i = 0
    for c in range(img.c):
        for h in range(img.h):
            for k in range(img.w):
                data[0][c][h][k] = img.data[i]
                i = i + 1

    (mod, params) = _get_tvm_output(net, data, build_dtype)
    return (mod, params)

def _get_tvm_output(net, data, build_dtype='float32', states=None):
    '''Compute TVM output'''
    dtype = 'float32'
    mod, params = relay.frontend.from_darknet(net, data.shape, dtype)
    return (mod, params)

def test_tidl_yolo():
    model_name = 'yolov3'
    cfg_name = model_name + '.cfg'
    weights_name = model_name + '.weights'
    cfg_url = 'https://github.com/pjreddie/darknet/blob/master/cfg/' + cfg_name + '?raw=true'
    weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'
    net = _load_net(cfg_url, cfg_name, weights_url, weights_name)
    build_dtype = {}
    front_out = verify_darknet_frontend(net, build_dtype)

    mod = front_out[0]
    LIB.free_network(net)

    print('---------- Original Graph ----------')
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    print(mod.astext(show_meta_data=False))
    print('---------- Merge Composite Functions ----------')
    mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod) #Merge sequence of ops into composite functions/ops
    print(mod.astext(show_meta_data=False))
    print("---------- Annotated Graph ----------")
    mod = transform.AnnotateTarget("tidl")(mod) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
    print(mod.astext(show_meta_data=False))
    print("---------- Merge Compiler Regions ----------")
    mod = transform.MergeCompilerRegions()(mod) #Merge annotated regions together that use the same external target, combines marked regions for each target
    print(mod.astext(show_meta_data=False))
    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    print(mod.astext(show_meta_data=False))
    print("---------- Pruned Graph ----------")
    mod = PruneSubgraphs(mod, compiler="tidl", num_subgraphs_to_keep=4)
    print(mod.astext(show_meta_data=False))
"""

from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.function import Function

class VarReplacer(ExprMutator):
    def __init__(self, var_map):
        ExprMutator.__init__(self)
        self.var_map = var_map

    def visit_var(self, var):
        if var in self.var_map:
            return self.var_map[var]
        return super().visit_var(var)

def UnpackComposites(mod, compiler="tidl"):
    class Unpacker(ExprMutator):
        def __init__(self):
            ExprMutator.__init__(self)

        def visit_call(self, call):
            if isinstance(call.op, Function):
                if call.op.attrs and call.op.attrs['Composite'] != "":
                    # unpack the function back into new main function.
                    var_map = {}
                    for arg, param in zip(call.args, call.op.params):
                        var_map[param] = super().visit(arg)
                    return VarReplacer(var_map).visit(call.op.body)
            return super().visit_call(call)
    
    for func in mod.get_global_vars():
        name = func.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != compiler:
            continue
        mod[name] = Unpacker().visit(mod[name])
    return mod

#TODO: move inside tidl import 
def generate_subgraph_tensors(mod, params, input_node, input_data):
    """
    """

    # From partitioned module, create a "calibration model" which can be
    # executed on CPU and will give additional outputs for boundary tensors.
    mod_tvm = relay.transform.InferType()(mod)
    mod_tvm = relay.transform.Inline()(mod_tvm)
    my_mutator = tidl.CalibrationGraphMutator("tidl")
    mod_tvm["main"] = my_mutator.make_calibration_graph(mod_tvm["main"])
    print("---------- Calibration module: ---------", mod_tvm)
    #print(mod_tvm)
    print("Input/output map:", my_mutator.name_map)

    # Build and execute calibration graph to get outputs
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    mod.set_input(input_node, input_data)
    mod.set_input(**params)
    mod.run()
    #mod.run(data=input_data, weight1=params_w1, weight2=params_w2)

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    #num_outputs = mod.get_num_outputs()
    #out = mod.get_output(0).asnumpy()
    #for i in range(num_outputs-len(my_mutator.name_map)-1):
    #    out = np.concatenate((out,mod.get_output(i+1).asnumpy()))
    #np.savetxt('graph_output.txt', out.flatten(), fmt='%10.5f')
    #print("Number of output tensors: " + str(num_outputs-len(my_mutator.name_map)))

    # We now have subgraph inputs
    # {1: 'tidl_1_i0', 2: 'tidl_1_o0', 3: 'tidl_0_i0', 4: 'tidl_0_o0'}
    subgraph_tensors = {}
    for i in range(len(results)):
        if i in my_mutator.name_map:
            subgraph_tensors[my_mutator.name_map[i]]=results[i]
            file_name = my_mutator.name_map[i] + ".txt"
            #np.savetxt(file_name, results[i].flatten(), fmt='%10.5f')
        else: # full graph output
            file_name = "graph_out_" + str(i) + ".txt"
            np.savetxt(file_name, results[i].flatten(), fmt='%10.5f')

    for key, value in subgraph_tensors.items():
        print("Subgraph tensor: ", key, value.shape)

    return subgraph_tensors

def test_mxnet_mobilenet_ssd():

    #model = 'ssd_512_mobilenet1.0_coco'
    model = 'ssd_512_mobilenet1.0_voc'
    image_size = 512
    layout = "NCHW"

    input_name = 'data'
    input_shape = (1, 3, image_size, image_size)
    dtype = 'float32'
    x, img = data.transforms.presets.ssd.load_test(im_fname, short=image_size)
    block = model_zoo.get_model(model, pretrained=True)

    mod_tvm, params_tvm = relay.frontend.from_mxnet(block, {"data": input_shape})

    #======================== Execute the full graph on TVM ====================
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod_tvm, "llvm", params=params_tvm)
    mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    #tvm_input = tvm.nd.array(x.asnumpy(), ctx=tvm.cpu(0))
    tvm_input = x.asnumpy()
    np.save("ssd_input.npy",tvm_input)
    mod.set_input(input_name, tvm_input)
    mod.set_input(**params)
    mod.run()
    class_IDs, scores, bounding_boxs = mod.get_output(0), mod.get_output(1), mod.get_output(2)
    ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                         class_IDs.asnumpy()[0], class_names=block.classes)
    plt.savefig("gluoncv_ssd_tvm.png")

    results = [mod.get_output(i).asnumpy() for i in range(mod.get_num_outputs())]
    print("Number of outputs: " + str(len(results)))
    for i in range(len(results)):
        np.savetxt("graph_out_"+str(i)+".txt", results[i].flatten(), fmt='%10.5f')


    block.hybridize()
    block.forward(x)
    block.export('temp')    # create file temp-symbol.json

#    model_json = mx.symbol.load('temp-symbol.json')
#    save_dict = mx.ndarray.load('temp-0000.params')
#    arg_params = {}
#    aux_params = {}
#    for k, v in save_dict.items():
#        tp, name = k.split(':', 1)
#        if tp == 'arg':
#            arg_params[name] = v
#        elif tp == 'aux':
#            aux_params[name] = v
#    mod, params = relay.frontend.from_mxnet(model_json, {input_name: input_shape}, arg_params=arg_params, aux_params=aux_params)

    mod = mod_tvm
    params = params_tvm
    print('---------- Original Graph ----------')
    mod = relay.transform.RemoveUnusedFunctions()(mod)
    #print(mod.astext(show_meta_data=False))
    mod['main'] = bind_params_by_name(mod['main'], params)
    mod0 = mod

    print('---------- Merge Composite Functions ----------')
    mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod) #Merge sequence of ops into composite functions/ops
    #print(mod.astext(show_meta_data=False))
    print("---------- Annotated Graph ----------")
    mod = transform.AnnotateTarget("tidl")(mod) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
    #print(mod.astext(show_meta_data=False))
    print("---------- Merge Compiler Regions ----------")
    mod = transform.MergeCompilerRegions()(mod) #Merge annotated regions together that use the same external target, combines marked regions for each target
    #print(mod.astext(show_meta_data=False))
    print("---------- Partioned Graph ----------")
    mod = transform.PartitionGraph()(mod)
    #print(mod.astext(show_meta_data=False))
    mod = UnpackComposites(mod, "tidl")
    print("---------- Pruned Graph ----------")
    mod = PruneSubgraphs(mod, compiler="tidl", num_subgraphs_to_keep=1)
    print(mod.astext(show_meta_data=False))

    #============= Generate subgraph boundary tensors ==============
    #input_data = np.expand_dims(img.transpose(2,0,1), axis=0)
    #input_data = x
    #img_norm   = (img-128.0)/128.0
    #np.save("street_small_tmp.npy",img)
    #input_data = np.expand_dims(img_norm.transpose(2,0,1), axis=0)
    subgraph_tensors = generate_subgraph_tensors(mod, params, input_name, tvm_input)

    #======================== Import the graph to TIDL ========================
    if tidl.import_relay_ir(mod, params, subgraph_tensors, layout, tidl_calib_tool, artifacts_folder) == True:
        print('Heterogeneous execution with TIDL.')
        graph, lib, params = relay.build_module.build(mod, target=target, params=params)
    else:
        print("Full graph compilation with LLVM.")
        # Future optimization: if not all subgraphs failed with TIDL import, re-partition
        # the graph to have only TIDL subgraphs with successful TIDL import. 
        graph, lib, params = relay.build_module.build(mod0, target=target, params=params)

    path_lib    = artifacts_folder + "deploy_lib.so"
    path_graph  = artifacts_folder + "deploy_graph.json"
    lib.export_library(path_lib, cc=arm_gcc) # for heterogeneous execute on TIDL+ARM
    path_params = artifacts_folder + "deploy_param.params"

    with open(path_graph, "w") as fo:
      fo.write(graph)
    with open(path_params, "wb") as fo:
      fo.write(relay.save_param_dict(params))


def test_gluoncv_segmentation():
    model_list = {
        #'deeplab_resnet101_coco': 'deeplab_resnet101_coco',
        #'deeplab_resnet50_ade':'deeplab_resnet50_ade',
        #'fcn_resnet50_ade' : 'fcn_resnet50_ade',
        #'fcn_resnet101_ade' : 'fcn_resnet101_ade',
        #'psp_resnet50_ade' : 'psp_resnet50_ade',
        #'psp_resnet101_ade' : 'psp_resnet101_ade',
        #'fcn_resnet101_coco' : 'fcn_resnet101_coco',
        #'psp_resnet101_coco' : 'psp_resnet101_coco',
        #'fcn_resnet101_voc' : 'fcn_resnet101_voc',
        #'psp_resnet101_voc' : 'psp_resnet101_voc',
        #'psp_resnet101_citys' : 'psp_resnet101_citys',
        'mask_rcnn_resnet18_v1b_coco' : 'mask_rcnn_resnet18_v1b_coco', 
        #'mask_rcnn_fpn_resnet18_v1b_coco' : 'mask_rcnn_fpn_resnet18_v1b_coco', 
        #'mask_rcnn_resnet50_v1b_coco' : 'mask_rcnn_resnet50_v1b_coco', 
        #'mask_rcnn_fpn_resnet50_v1b_coco' : 'mask_rcnn_fpn_resnet50_v1b_coco',        
        #'mask_rcnn_resnet101_v1d_coco' : 'mask_rcnn_resnet101_v1d_coco',
        #'mask_rcnn_fpn_resnet101_v1d_coco' : 'mask_rcnn_fpn_resnet101_v1d_coco',
    }

    for model_name in model_list:
        print(model_name)
        model = gluoncv.model_zoo.get_model(model_name, pretrained=True)
        url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'
        filename = 'example.jpg'
        gluoncv.utils.download(url, filename)
        img = image.imread(filename)
        img = test_transform(img, ctx = mx.cpu(0))

        input_name = "data"
        input_shape = img.shape
        layout = "NCHW"

        tvm_input = img.asnumpy()
        np.save("seg_input.npy",tvm_input)
        model.hybridize()
        model.forward(img)
        model.export('gluoncv-temp')    # create file gluoncv-temp-symbol.json

        mod, params = relay.frontend.from_mxnet(model, {input_name:input_shape})
        mod0 = mod
        print('---------- Original Graph ----------')
        mod = relay.transform.RemoveUnusedFunctions()(mod)
        mod['main'] = bind_params_by_name(mod['main'], params)
        print(mod.astext(show_meta_data=False))
        print('---------- Merge Composite Functions ----------')
        mod = tvm.relay.op.contrib.tidl._merge_sequential_ops(mod) #Merge sequence of ops into composite functions/ops
        #print(mod.astext(show_meta_data=False))
        print("---------- Annotated Graph ----------")
        mod = transform.AnnotateTarget("tidl")(mod) #Looks at annotated ops and marks them in the graph with compiler.begin and compiler.end
        #print(mod.astext(show_meta_data=False))
        print("---------- Merge Compiler Regions ----------")
        mod = transform.MergeCompilerRegions()(mod) #Merge annotated regions together that use the same external target, combines marked regions for each target
        #print(mod.astext(show_meta_data=False))
        print("---------- Partioned Graph ----------")
        mod = transform.PartitionGraph()(mod)
        mod = UnpackComposites(mod, "tidl")
        #print(mod.astext(show_meta_data=False))
        print("---------- Pruned Graph ----------")
        mod = PruneSubgraphs(mod, compiler="tidl", num_subgraphs_to_keep=1)
        print(mod.astext(show_meta_data=False))
        subgraph_tensors = generate_subgraph_tensors(mod, params, input_name, tvm_input)

        #======================== Import the graph to TIDL ========================
        if tidl.import_relay_ir(mod, params, subgraph_tensors, layout, tidl_calib_tool, artifacts_folder) == True:
            print('Heterogeneous execution with TIDL.')
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
        else:
            print("Full graph compilation with LLVM.")
            # Future optimization: if not all subgraphs failed with TIDL import, re-partition
            # the graph to have only TIDL subgraphs with successful TIDL import. 
            graph, lib, params = relay.build_module.build(mod0, target=target, params=params)
    
        path_lib    = artifacts_folder + "deploy_lib.so"
        path_graph  = artifacts_folder + "deploy_graph.json"
        lib.export_library(path_lib, cc=arm_gcc) # for heterogeneous execute on TIDL+ARM
        path_params = artifacts_folder + "deploy_param.params"
    
        with open(path_graph, "w") as fo:
          fo.write(graph)
        with open(path_params, "wb") as fo:
          fo.write(relay.save_param_dict(params))

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

    artifacts_folder = "./artifacts/"
    if os.path.isdir(artifacts_folder):
        filelist = [ f for f in os.listdir(artifacts_folder)]
        for file in filelist:
            os.remove(os.path.join(artifacts_folder, file))
    else:
        os.mkdir(artifacts_folder)

    target = "llvm -target=armv7l-linux-gnueabihf"

    #test_tidl_annotation()
    #test_tidl_mobilenet()
    #test_tidl_mobilenet_no_composite()
    #test_tidl_yolo()
    #test_mxnet_mobilenet_ssd()
    #test_extern_tidl()
    test_gluoncv_segmentation()
