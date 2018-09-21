from __future__ import absolute_import as _abs
import numpy as np
import tvm
import nnvm.compiler
from nnvm.frontend.caffe2 import from_caffe2
from tvm.contrib import graph_runtime

from caffe2.python import workspace
from caffe2.proto import caffe2_pb2

from PIL import Image

from timeit import default_timer as timer

def load_caffe2_model(init, predict):
    init_net = caffe2_pb2.NetDef()
    with open(init, 'rb') as f:
        init_net.ParseFromString(f.read())

    predict_net = caffe2_pb2.NetDef()
    with open(predict, 'rb') as f:
        predict_net.ParseFromString(f.read())
    return init_net, predict_net

def save_picture(fname, output):
    result = Image.fromarray(np.uint8((output[0, 0]).clip(0, 255)), mode='L')
    canvas = np.full((192, 192), 255)
    canvas[:, :] = np.asarray(result)
    plt.imshow(canvas.astype(np.uint8))
    plt.savefig(fname)

def run_tvm_experiment(init_net, predict_net, input):
    print 'Running TVM version'
    start = timer()


    sym, params = from_caffe2(init_net, predict_net)

    # assume first input name is data
    input_name = sym.list_input_names()[0]
    shape_dict = {input_name: input.shape}

    opt_level = 2
    target = 'llvm -mcpu=core-avx2'
    graph = None
    lib = None
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            sym, target, shape={input_name: input.shape}, params=params)

    ctx = tvm.context(target, 0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_name, tvm.nd.array(input.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    output_shape = (1000,1000,1000,1000)
    tvm_output = m.get_output(0).asnumpy()

    end = timer()
    print 'TIME:', end-start
    return tvm_output

def run_c2_experiment(init_net, predict_net, input):
    print 'Running Caffe2 version'
    start = timer()
    workspace.RunNetOnce(init_net)

    workspace.FeedBlob("data", input.astype('float32'))


    workspace.RunNetOnce(predict_net)
    output_blob = predict_net.external_output[0]

    c2_output = workspace.FetchBlob(output_blob)
    end = timer()
    print 'TIME:', end-start
    return c2_output

def main():
    img = Image.open('cat.png').resize((224, 224))

    image = np.asarray(img)
    image = image.transpose((2, 0, 1))
    x = image[np.newaxis, :]

    init_net, predict_net = load_caffe2_model('squeeze_netv1.1_init_net.pb', 'squeeze_netv1.1_predict_net.pb')

    tvm_output = run_tvm_experiment(init_net, predict_net, x)
    print np.argmax(tvm_output)

    c2_output = run_c2_experiment(init_net, predict_net, x)
    print np.argmax(c2_output)

main()
