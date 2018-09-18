"""Caffe2 frontend"""
from __future__ import absolute_import as _abs
import numpy as np
import tvm
from nnvm import symbol as _sym
from nnvm import graph as _graph
from nnvm.compiler import graph_util
from nnvm.frontend.common import get_nnvm_op, Renamer, SymbolTable, AttrConverter as AttrCvt

from caffe2.python import workspace
from caffe2.proto import caffe2_pb2


def _dimension_picker(prefix, surfix=''):

    def _impl(arg):
        if 'kernels' in arg:
            kernel = arg['kernels']
            if len(kernel) == 2:
                return prefix + '2d' + surfix
            else:
                raise NotImplementedError("Only 2d kernel supported.")
        elif 'kernel' in arg:
            return prefix + '2d' + surfix

    return _impl

def _dimension_constraint():

    def _dim_check(arg):
        if 'kernel' in arg or ('kernels' in arg and len(arg['kernels']) == 2):
            return True
        return False

    return _dim_check, "Only 2d kernel supported."

def _infer_channels(inputs, params, transpose=False):
    """A hack for getting 'channels' or 'units' since onnx don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels

def _revert_caffe2_pad(pads):
    """Caffe2 require two times the normal padding."""
    if len(pads) == 4:
        pads = pads[:2]
    elif len(pads) == 2:
        pass
    else:
        raise ValueError("Invalid caffe2 type padding: {}".format(pads))
    return pads




class Caffe2OpConverter(object):
    """ A helper class for holding Caffe2 op converters.
    """

    @classmethod
    def get_converter(cls):
        """ Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, '_impl'):
            return getattr(cls, '_impl')
        else:
            raise NotImplementedError(
                '{} not implemented'.format(cls.__name__))

class Pool(Caffe2OpConverter):
    """ A helper class for pool op converters.
    """

    name = ''

    @classmethod
    def _impl(cls, inputs, args, params):
        if 'global_pooling' in args and args['global_pooling'] == 1:
            return AttrCvt(
                op_name=_dimension_picker('global_' + cls.name),
                ignores={
                    'float16_compute',
                    'legacy_pad',
                    'global_pooling',
                    'shared_buffer',
                    'dilations',
                    'kernel', 'kernel_h', 'kernel_w',
                    'stride', 'stride_h', 'stride_w',
                    'dilation', 'dilation_h', 'dilation_w',
                    'pad', 'pad_t', 'pad_l', 'pad_b', 'pad_r',
                    'order',
                    'cudnn_exhaustive_search',
                },
                custom_check=_dimension_constraint())(inputs, args, params)
        else:
            if 'stride_h' in args or 'stride_w' in args:
                assert 'stride_h' in args and 'stride_w' in args
                assert 'stride' not in args and 'strides' not in args
                args['strides'] = [args['stride_h'], args['stride_w']]
                args.pop('stride_h')
                args.pop('stride_w')
            return AttrCvt(
                op_name=_dimension_picker(cls.name),
                transforms={
                    'kernels': 'pool_size',
                    'kernel': ('pool_size', (1, 1), lambda x: [x, x]),
                    'pads': ('padding', (0, 0), _revert_caffe2_pad),
                    'pad' : ('padding', (0, 0), lambda x: [x, x]),
                    'group': ('groups', 1),
                    'strides':'strides',
                    'stride' : ('strides', (1, 1), lambda x: [x, x]),
                },
                excludes={
                    'kernel_h', 'kernel_w',
                    'dilation', 'dilation_h', 'dilation_w',
                    'pad_t', 'pad_l', 'pad_b', 'pad_r',
                },
                ignores={
                    'float16_compute',
                    'global_pooling',
                    'legacy_pad',
                    'shared_buffer',
                    'dilations',
                    'order',
                    'cudnn_exhaustive_search',
                    'init_params',
                },
                custom_check=_dimension_constraint())(inputs, args, params)

class AveragePool(Pool):
    name = 'avg_pool'

class MaxPool(Pool):
    name = 'max_pool'

class Conv(Caffe2OpConverter):
    @classmethod
    def _impl(cls, inputs, args, params):
        # get number of channels
        channels = _infer_channels(inputs[1], params)
        args['channels'] = channels
        #TODO: do the same for padding and dilation
        if 'stride_h' in args or 'stride_w' in args:
            assert 'stride_h' in args and 'stride_w' in args
            assert 'stride' not in args and 'strides' not in args
            args['strides'] = [args['stride_h'], args['stride_w']]
            args.pop('stride_h')
            args.pop('stride_w')

        return AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernels': 'kernel_size',
                'kernel': ('kernel_size', (1, 1), lambda x: [x, x]),
                'dilations': ('dilation', (0, 0)),
                'pads': ('padding', (0, 0), _revert_caffe2_pad),
                'pad' : ('padding', (0, 0), lambda x: [x, x]),
                'group': ('groups', 1),
                'strides':'strides',
                'stride' : ('strides', (1, 1), lambda x: [x, x]),
                'order': ('layout', ("NCHW"), lambda x: x if isinstance(x, str) else x.decode('UTF-8')),
            },
            excludes={
                'legacy_pad',
                'global_pooling',
                'kernel_h', 'kernel_w',
                'dilation', 'dilation_h', 'dilation_w',
                'pad_t', 'pad_l', 'pad_b', 'pad_r',
            },
            ignores={
                'float16_compute',
                'shared_buffer',
                'convolution_transform_strategy',
                'exhaustive_search',
                'algo',
                'init_params',
                'cudnn_exhaustive_search',
                'adj',
                'hwgq',
            },
            extras={'use_bias': len(inputs) == 3},
            custom_check=_dimension_constraint())(inputs, args, params)

class Concat(Caffe2OpConverter):
    @classmethod
    def _impl(cls, inputs, args, params):
        def _get_axis_from_order_str(order):
            order = order if isinstance(order, str) else order.decode('UTF-8')
            if order == 'NCHW':
                return 1
            elif order == 'NHWC':
                return 3
            else:
                raise RuntimeError("Unsupported storage order: {} in caffe2", order)

        return AttrCvt(
            op_name='concatenate',
            transforms={
                'order' : ('axis', (1), _get_axis_from_order_str),
            },
            excludes={
                'add_axis',
            })(inputs, args, params)

class Softmax(Caffe2OpConverter):

    @classmethod
    def _impl(cls, inputs, args, params):
        if 'axis' not in args:
            args['axis'] = 1
        return AttrCvt(
            op_name='softmax',
            transforms={
                 'axis': ('axis', 1),
            })(inputs, args, params)

class NormalizePlanarYUV(Caffe2OpConverter):
    @classmethod
    def _impl(cls, inputs, args, params):
        assert len(inputs) == 3
        X = inputs[0]
        M = inputs[1]
        S = inputs[2]

        M_name = M.attr('name')
        params[M_name] = tvm.nd.array(np.array(params[M_name].asnumpy()).reshape([1,3,1,1]))
        S_name = S.attr('name')
        params[S_name] = tvm.nd.array(np.array(params[S_name].asnumpy()).reshape([1,3,1,1]))

        return _sym.broadcast_div(_sym.broadcast_sub(X, M), S)

class ResizeNearest(Caffe2OpConverter):
    """ Operator converter for Upsample (nearest mode).
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        width_scale = args['width_scale'] if 'width_scale' in args else 1
        height_scale = args['height_scale'] if 'height_scale' in args else 1
        assert width_scale == height_scale

        return _sym.upsampling(inputs[0], scale=int(width_scale), method="NEAREST_NEIGHBOR")

class Elemwise(Caffe2OpConverter):
    """ A helper class for elemwise op converters.
    """

    name = ''

    @classmethod
    def _math_name_picker(cls, suffix):

        def _impl(attr):
            if attr.get('broadcast', 0):
                return 'broadcast_' + suffix
            return 'elemwise_' + suffix

        return _impl

    @classmethod
    def _impl(cls, inputs, attr, params):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(
            len(inputs))
        op_name = cls._math_name_picker(cls.name)(attr)
        axis = int(attr.get('axis', 0))
        conv_ops = ["conv2d", "conv2d_transpose"]
        if op_name == 'broadcast_add' and inputs[0].attr('op_name') in conv_ops:
            inputs[1] = _sym.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_nnvm_op(op_name)(*inputs)

class Add(Elemwise):
    name = 'add'

# compatible operators that do NOT require any conversion.
_identity_list = []


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
# Minimal set of ops for squeezenet
def _get_convert_map():
    return {
        'Add': Add.get_converter(),
        'Relu': AttrCvt('relu', {}, ignores=['order']),
        'Sigmoid': Renamer('sigmoid'),

        'Softmax': Softmax.get_converter(),

        # nn
        'AveragePool': AveragePool.get_converter(),
        'MaxPool': MaxPool.get_converter(),
        'Conv': Conv.get_converter(),
        'Dropout': AttrCvt('dropout', {'ratio': 'rate'}, ignores=['is_test']),
        'Concat': Concat.get_converter(),

        # c2 image preprocessing off
        'NormalizePlanarYUV': NormalizePlanarYUV.get_converter(),
        'ResizeNearest': ResizeNearest.get_converter(),
    }


class Caffe2_NetDef(object):
    """A helper class for handling nnvm graph copying from pb2.GraphProto.
    Definition: https://todo
    """

    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._external_inputs = set()
        self._visitedNodes = set()
        self._defOp = {}

    def from_caffe2(self, init_net, predict_net):
        """

        Parameters
        ----------
        workspace : Caffe2 workspace
        predict_net : protobuf object

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """

        workspace.RunNetOnce(init_net)

        shapes = {}
        for op in init_net.op:
            for blob in op.output:
                args = self._parse_arg(op.arg)
                if 'shape' in args:
                    shapes[blob] = args['shape']

        # Params
        self._params = {}
        used_blobs = set()
        for c2_op in predict_net.op:
            for i in c2_op.input:
                used_blobs.add(i)
        for blob in workspace.Blobs():
            if blob in used_blobs and blob != 'data':
                self._params[blob] = tvm.nd.array(workspace.FetchBlob(blob))

        # Variables
        self._nodes = {}
        for blob in predict_net.external_input:
            self._nodes[blob] = _sym.Variable(name=blob)

        # Ops
        for c2_op in predict_net.op:
            for blob in c2_op.output:
                self._defOp[blob] = c2_op
        for c2_op in predict_net.op:
            self._processOp(c2_op)

        # Outputs
        out = []
        for blob in predict_net.external_output:
            out.append(self._nodes[blob])

        if len(out) > 1:
            sym = _sym.Group(out)
        else:
            sym = out[0]

        return sym, self._params

    def _getNode(self, blob):
        if blob in self._nodes:
            return self._nodes[blob]

        assert not blob in self._visitedNodes, 'Cyclic dependency in the graph (in {})'.format(blob)
        self._visitedNodes.add(blob)

        self._processOp(self._defOp[blob])
        return self._nodes[blob]

    def _processOp(self, c2_op):
        op_type = c2_op.type
        args = self._parse_arg(c2_op.arg)
        inputs = [self._getNode(i) for i in c2_op.input]
        tvm_op = self._convert_operator(op_type, inputs, args)
        # Ignore all outputs except the first one
        self._nodes[c2_op.output[0]] = tvm_op[0]

    def _parse_arg(self, arg):
        """Convert a list of Argument to a dict, with names as keys."""
        args = {}
        for a in arg:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    args[a.name] = getattr(a, f)
            for f in ['floats', 'ints', 'strings']:
                if list(getattr(a, f)):
                    assert a.name not in args, "Only one type of attr is allowed"
                    args[a.name] = tuple(getattr(a, f))
            for f in ['n']:
                if a.HasField(f):
                    raise NotImplementedError(
                        "Field {} is not supported in nnvm.".format(f))
            for f in ['nets']:
                if list(getattr(a, f)):
                    raise NotImplementedError(
                        "Field {} is not supported in nnvm.".format(f))
            if a.name not in args:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return args

    def _convert_operator(self,
                          op_type,
                          inputs,
                          args,
                          identity_list=None,
                          convert_map=None):
        """Convert from Caffe2 operator to nnvm operator.
        The converter must specify conversions explicity for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_type : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of nnvm.Symbol
            List of input symbols.
        args : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to nnvm, callable are functions which
            take args and return (new_op_type, new_args)

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _get_convert_map()
        if op_type in identity_list:
            sym = get_nnvm_op(op_type)(*inputs, **args)
        elif op_type in convert_map:
            # Add a sanitizing step to convert all byte strings in args to strings
            sym = convert_map[op_type](inputs, args, self._params)
        else:
            raise NotImplementedError(
                "Operator {} not implemented.".format(op_type))
        return sym

def from_caffe2(init_net, predict_net):
    """
    TODO: Fill in description

    Parameters
    ----------
    init_net : protobug_object
        Caffe2 ModelProto
    predict_net : protobug_object
        Caffe2 ModelProto

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """

    c2 = Caffe2_NetDef()
    return c2.from_caffe2(init_net, predict_net)
