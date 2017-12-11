# -*- coding: utf-8 -*-
# -*- encoding:utf-8 -*-
__author__ = 'Han Wang'
import os.path
import pdb
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn_cell_impl import _Linear
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

LSTMStateList=collections.namedtuple('LSTMStateList',('t','c','h'))
LSTMStateTuple=collections.namedtuple('LSTMStateTuple',('c','h'))

class decoderLSTM(LSTMCell):
	def __init__(self, num_units,
		use_peepholes=False, cell_clip=None,
		initializer=None, num_proj=None, proj_clip=None,
		forget_bias=1.0,
		activation=None, reuse=None):

		"""
		Initialize the parameters for an LSTM cell.
		Args:
			num_units: int, The number of units in the LSTM cell.
			use_peepholes: bool, set True to enable diagonal/peephole connections.
			cell_clip: (optional) A float value, if provided the cell state is clipped
			by this value prior to the cell output activation.
			initializer: (optional) The initializer to use for the weight and
			projection matrices.
			num_proj: (optional) int, The output dimensionality for the projection
			matrices.  If None, no projection is performed.
			proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
			provided, then the projected values are clipped elementwise to within
			`[-proj_clip, proj_clip]`.
			num_unit_shards: Deprecated, will be removed by Jan. 2017.
			Use a variable_scope partitioner instead.
			num_proj_shards: Deprecated, will be removed by Jan. 2017.
			Use a variable_scope partitioner instead.
			forget_bias: Biases of the forget gate are initialized by default to 1
			in order to reduce the scale of forgetting at the beginning of
			the training. Must set it manually to `0.0` when restoring from
			CudnnLSTM trained checkpoints.
			state_is_tuple: If True, accepted and returned states are 2-tuples of
			the `c_state` and `m_state`.  If False, they are concatenated
			along the column axis.  This latter behavior will soon be deprecated.
			activation: Activation function of the inner states.  Default: `tanh`.
			reuse: (optional) Python boolean describing whether to reuse variables
			in an existing scope.  If not `True`, and the existing scope already has
			the given variables, an error is raised.

		When restoring from CudnnLSTM-trained checkpoints, must use
		CudnnCompatibleLSTMCell instead.

		"""

		super(LSTMCell, self).__init__(_reuse=reuse)

		self._num_units = num_units
		self._use_peepholes = use_peepholes
		self._cell_clip = cell_clip
		self._initializer = initializer
		self._num_proj = num_proj
		self._proj_clip = proj_clip
		self._forget_bias = forget_bias
		self._activation = activation or math_ops.tanh

		self._state_size=LSTMStateList(num_units,num_units,num_units)
		self._output_size = num_units
		self._linear1 = None
		self._linear2 = None
		self._linear3=None

	@property
	def state_size(self):
		return self._state_size

	def call(self,inputs,state):
		"""Long short-term memory cell (LSTM).

	    Args:
	      inputs: `2-D` tensor with shape `[batch_size x input_size]`. ht
	      state: An `LSTMStateTuple` of state tensors, each shaped
	        `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
	        `True`.  Otherwise, a `Tensor` shaped
	        `[batch_size x 2 * self.state_size]`. 

	    Returns:
	      A pair containing the new hidden state, and the new state (either a
	        `LSTMStateTuple` or a concatenated state, depending on
	        `state_is_tuple`).
	    """

		sigmoid = math_ops.sigmoid
		tanh=math_ops.tanh
		# Parameters of gates are concatenated into one multiply for efficiency.
		(t, c, h) = state
		with tf.variable_scope('ifz',initializer=self._initializer):
			self._linear1=_Linear([inputs, h, t], 3*self._num_units, True) #i,f,z
			i,f,z=array_ops.split(
	    		value=self._linear1([inputs, h, t]), num_or_size_splits=3, axis=1)
		new_c=sigmoid(f)*c+sigmoid(i)*tanh(z)
		with tf.variable_scope('o',initializer=self._initializer):
			self._linear2=_Linear([inputs, h, new_c], self._num_units, True) #o
			o=self._linear2([inputs, h, new_c])
		new_h=sigmoid(o)*tanh(new_c)
		with tf.variable_scope('t',initializer=self._initializer):
			self._linear3=_Linear([new_h], self._num_units, True) #t
			t=self._linear3([new_h])
		return new_h,LSTMStateList(t,new_c,new_h)

class encoderLSTM(LSTMCell):
	def __init__(self, num_units,
		use_peepholes=False, cell_clip=None,
		initializer=None, num_proj=None, proj_clip=None,
		forget_bias=1.0,
		activation=None, reuse=None):
		super(LSTMCell, self).__init__(_reuse=reuse)

		self._num_units = num_units
		self._use_peepholes = use_peepholes
		self._cell_clip = cell_clip
		self._initializer = initializer
		self._num_proj = num_proj
		self._proj_clip = proj_clip
		self._forget_bias = forget_bias
		self._activation = activation or math_ops.tanh

		self._state_size=LSTMStateTuple(num_units,num_units)
		self._output_size = num_units
		self._linear1 = None
		self._linear2 = None
		self._linear3=None

	@property
	def state_size(self):
		return self._state_size

	def call(self,inputs,state):
		sigmoid = math_ops.sigmoid
		tanh=math_ops.tanh
		# Parameters of gates are concatenated into one multiply for efficiency.
		(c, h) = state
		with tf.variable_scope('if',initializer=self._initializer):
			self._linear1=_Linear([inputs, h, c], 2*self._num_units, True) #i,f
			i,f=array_ops.split(
	    		value=self._linear1([inputs, h, c]), num_or_size_splits=2, axis=1)
		with tf.variable_scope('z',initializer=self._initializer):
			self._linear2=_Linear([inputs,h],self._num_units,True) #z
			z=self._linear2([inputs,h])
		new_c=sigmoid(f)*c+sigmoid(i)*tanh(z)
		with tf.variable_scope('o',initializer=self._initializer):
			self._linear2=_Linear([inputs, h, new_c], self._num_units, True) #o
			o=self._linear2([inputs, h, new_c])
		new_h=sigmoid(o)*tanh(new_c)
		return new_h,LSTMStateTuple(new_c,new_h)
