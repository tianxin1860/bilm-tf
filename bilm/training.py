'''
Train and test bidirectional language models.
'''

import os
import time
import json
import re
import pickle

import tensorflow as tf
import numpy as np

DTYPE = 'float32'
DTYPE_INT = 'int64'

tf.logging.set_verbosity(tf.logging.INFO)
name_dict = {
  'lm/embedding/embedding:0':1,
  'lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0':21,
  'lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0':22,
  'lm/RNN_0/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel:0':23,
  'lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0':31,
  'lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0':32,
  'lm/RNN_0/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel:0':33,
  'lm/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0':41,
  'lm/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0':42,
  'lm/RNN_1/rnn/multi_rnn_cell/cell_0/lstm_cell/projection/kernel:0':43,
  'lm/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0':51,
  'lm/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0':52,
  'lm/RNN_1/rnn/multi_rnn_cell/cell_1/lstm_cell/projection/kernel:0':53,
  'lm/RNN_0/rnn/lstm_cell/bias:0':21,
  'lm/RNN_0/rnn/lstm_cell/kernel:0':22,
  'lm/RNN_0/rnn/lstm_cell/projection/kernel:0':23,
  'lm/RNN_1/rnn/lstm_cell/bias:0':41,
  'lm/RNN_1/rnn/lstm_cell/kernel:0':42,
  'lm/RNN_1/rnn/lstm_cell/projection/kernel:0':43,

  'lm/softmax/b:0':61,
  'lm/softmax/W:0':62,
}

slot_dict = {}
def init_slot():
    global slot_dict
    slot_dict = {}

def name2slot(para_name, exact=False):
    res = []
    if exact:
        if para_name in name_dict:
            return [name_dict[para_name]]
        else:
            return []
    for key_name in name_dict.keys():
        if para_name.find(key_name) >= 0:
            res.append(name_dict[key_name])
    return res

def update_slot(slots, p_array):
    p_mean, p_max, p_min, p_num = p_array.mean(), p_array.max(), p_array.min(), np.prod(p_array.shape)
    for slot in slots:
        if slot in slot_dict:
            s_mean, s_max, s_min, s_num = slot_dict[slot]
            s_mean = (s_mean*s_num + p_mean*p_num) / (p_num + s_num)
            s_max = max(s_max, p_max)
            s_min = min(s_min, p_min)
            s_num = p_num + s_num
            slot_dict[slot] = [s_mean, s_max, s_min, s_num]
        else:
            slot_dict[slot] = [p_mean, p_max, p_min, p_num]

def record_slot(logger):
    for slot in slot_dict:
        logger.info("slot:" + "\t".join([str(round(x, 10)) for x in [slot] + slot_dict[slot]]))

def var_print(tag, p_array, p_name, name, logger, args):
    try:
        if isinstance(p_array,np.float32):
	    p_array=np.array([p_array]) 
        if not isinstance(p_array, np.ndarray):
	    p_array = p_array.values
        param_num = np.prod(p_array.shape)
    except:
        import pdb; pdb.set_trace()
        logger.info('var:{} {} type {} not surpported'.format(p_name, name, type(p_array)))
        return

    p_array3 = np.multiply(np.multiply(p_array, p_array), p_array)
    logger.info(tag + ": {0} ({1}),  l3={2} sum={3}  max={4}  min={5} mean={6} num={7} {8}".format(p_name, name, p_array3.sum(), p_array.sum(), p_array.max(), p_array.min(), p_array.mean(), p_array.shape, param_num))
    if args.detail:
	logger.info(" ".join([tag + "[", p_name, '] shape [', str(p_array.shape), ']', str(p_array)]))

def print_debug_info(sess, logger, vars_data=None, grad_data=None, grad_para_data=None, args=None):
    if not args.para_print:
	return
    if vars_data:
        vars, fetched_vars = vars_data
        for var, fetched_var in zip(vars, fetched_vars):
            shape = var.get_shape()
            p_array = fetched_var
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            var_print('var', p_array, var.name, var.name, logger, args)
    if grad_data:
        grad_vars, graded_vars = grad_data
        for grad, graded_var in zip(grad_vars, graded_vars):
            try:
                shape = grad.get_shape()
            except:
                logger.info('grad {} failed'.format(grad.name))
                continue
            p_array = graded_var
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            var_print('grad', p_array, grad.name, grad.name, logger, args)
    if grad_para_data:
        grad_vars, graded_vars = grad_para_data
        for grad, graded_var in zip(grad_vars, graded_vars):
            try:
                shape = grad.get_shape()
            except:
                logger.info('grad para {} failed'.format(grad.name))
                continue
            p_array = graded_var
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            var_print('grad para', p_array, grad.name, grad.name, logger, args)

    init_slot()
    total_parameters = 0
    parameters_string = ""
    
    for variable in tf.trainable_variables():
	shape = variable.get_shape()
	p_array = sess.run(variable.name)
	slots = name2slot(variable.name)
	if slots:
	    update_slot(slots, p_array)
	variable_parameters = 1
	for dim in shape:
	    variable_parameters *= dim.value
	total_parameters += variable_parameters
	var_print('para', p_array, variable.name, variable.name, logger, args)
    record_slot(logger)
    logger.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

def save_var(p_array, name, logger, args):
    if args.save_para_path:
        if name2slot(name, exact=True):
            new_name = 'slot_' + str(name2slot(name, exact=True)[0])
        else:
            new_name = name.replace('/', '%')
        with open(os.path.join(args.save_para_path, new_name + '.data'), 'wb') as fout:
            pickle.dump(p_array, fout)
            logger.info('saved {} to {}'.format(name, new_name))

def save_para(sess, logger, args=None):
    for variable in tf.trainable_variables():
	p_array = sess.run(variable.name)
	save_var(p_array, variable.name, logger, args)


class LanguageModel(object):
    '''
    A class to build the tensorflow computational graph for NLMs

    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.

    is_training is a boolean used to control behavior of dropout layers
        and softmax.  Set to False for testing.

    The LSTM cell is controlled by the 'lstm' key in options
    Here is an example:

     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},

        'projection_dim' is assumed token embedding size and LSTM output size.
        'dim' is the hidden state size.
        Set 'dim' == 'projection_dim' to skip a projection layer.
    '''
    def __init__(self, options, is_training, logger):
        self.options = options
        self.is_training = is_training
        self.logger = logger
        self.bidirectional = options.get('bidirectional', False)

        # for the loss function
        self.share_embedding_softmax = options.get(
            'share_embedding_softmax', False)

        self.sample_softmax = options.get('sample_softmax', False)

        self._build()

    def _build_word_embeddings(self):
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        projection_dim = self.options['lstm']['projection_dim']

        # the input token_ids and word embeddings
        self.token_ids = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids')
        # the word embeddings
        #with tf.device("/cpu:0"):
        with tf.variable_scope('embedding') as scope:
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                self.token_ids)

        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids_reverse')
            #with tf.device("/cpu:0"):
            with tf.variable_scope('embedding') as scope:
                self.embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.token_ids_reverse)


    def _build(self):
        # size of input options
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        self._build_word_embeddings()

        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)
        self.init_lstm_state = []
        self.final_lstm_state = []

        # get the LSTM inputs
        if self.bidirectional:
            self.lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            self.lstm_inputs = [self.embedding]

        # now compute the LSTM outputs
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        use_skip_connections = self.options['lstm'].get(
                                            'use_skip_connections')
        if use_skip_connections:
            self.logger.info("USING SKIP CONNECTIONS")
        if self.options['para_init']:
            init = tf.constant_initializer(self.options['init1'])
        else:
            init=None
        self.lstm_outputs = []
        self.lstm_unpack = []
        for lstm_num, lstm_input in enumerate(self.lstm_inputs):
            lstm_cells = []
            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        forget_bias=0.0, initializer=init)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # add dropout
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                        input_keep_prob=keep_prob)

                lstm_cells.append(lstm_cell)

            if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]

            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(
                    lstm_cell.zero_state(batch_size, DTYPE))
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                if self.bidirectional:
                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input, axis=1),
                            initial_state=self.init_lstm_state[-1])
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(lstm_input, axis=1),
                        initial_state=self.init_lstm_state[-1])
                self.final_lstm_state.append(final_state)

            # (batch_size * unroll_steps, 512)
            lstm_output_stack = tf.stack(_lstm_output_unpacked, axis=1)
            lstm_output_flat = tf.reshape(lstm_output_stack[:, :, 6:6+projection_dim], [-1, projection_dim])
            #lstm_output_flat = tf.reshape(lstm_output_stack, [-1, projection_dim])
            #lstm_output_flat = _lstm_output_unpacked
            if self.is_training:
                # add dropout to output
                lstm_output_flat = tf.nn.dropout(lstm_output_flat,
                    keep_prob)
            tf.add_to_collection('lstm_output_embeddings',
                _lstm_output_unpacked)

            self.lstm_unpack.append(lstm_output_stack)
            self.lstm_outputs.append(lstm_output_flat)

        self._build_loss(self.lstm_outputs)

    def _build_loss(self, lstm_outputs):
        '''
        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input

        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        n_tokens_vocab = self.options['n_tokens_vocab']

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps),
                                   name=name)
            return id_placeholder

        # get the window and weight placeholders
        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders(
                '_reverse')

        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax:
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        #with tf.variable_scope('softmax'), tf.device('/cpu:0'):
        with tf.variable_scope('softmax') as scope:
            # Glorit init (std=(1.0 / sqrt(fan_in))
            softmax_init = tf.random_normal_initializer(0.0,
                1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable(
                    'W', [n_tokens_vocab, softmax_dim],
                    dtype=DTYPE,
                    initializer=softmax_init
                )
            self.softmax_b = tf.get_variable(
                'b', [n_tokens_vocab],
                dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []
        self.losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs):
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                # get the full softmax loss
                output_scores = tf.matmul(
                    lstm_output_flat,
                    tf.transpose(self.softmax_W)
                ) + self.softmax_b
                # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                #   expects unnormalized output since it performs the
                #   softmax internally
                #losses = tf.reduce_mean(lstm_output_flat) + tf.reduce_mean(output_scores)

                #'''
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=output_scores,
                    labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                )
                #'''
                
                self.losses.append(losses)
            self.individual_losses.append(tf.reduce_mean(losses))

        # now make the total loss -- it's the mean of the individual losses
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0]
                                    + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]

def _get_feed_dict_from_X(X, start, end, model, bidirectional):
    feed_dict = {}
    token_ids = X['token_ids'][start:end]
    feed_dict[model.token_ids] = token_ids

    if bidirectional:
        feed_dict[model.token_ids_reverse] = \
            X['token_ids_reverse'][start:end]

    # now the targets with weights
    next_id_placeholders = [[model.next_token_id, '']]
    if bidirectional:
        next_id_placeholders.append([model.next_token_id_reverse, '_reverse'])

    for id_placeholder, suffix in next_id_placeholders:
        name = 'next_token_id' + suffix
        feed_dict[id_placeholder] = X[name][start:end]

    return feed_dict


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
    # wrapper around tf.clip_by_global_norm that also does summary ops of norms

    # compute norms
    # use global_norm with one element to handle IndexedSlices vs dense
    norms = [tf.global_norm([t]) for t in t_list]

    # summary ops before clipping
    summary_ops = []
    for ns, v in zip(norms, variables):
        name = 'norm_pre_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    # clip 
    clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

    # summary ops after clipping
    norms_post = [tf.global_norm([t]) for t in clipped_t_list]
    for ns, v in zip(norms_post, variables):
        name = 'norm_post_clip/' + v.name.replace(":", "_")
        summary_ops.append(tf.summary.scalar(name, ns))

    summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

    return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, options, do_summaries, global_step):
    # grads = [(grad1, var1), (grad2, var2), ...]
    def _clip_norms(grad_and_vars, val, name):
        # grad_and_vars is a list of (g, v) pairs
        grad_tensors = [g for g, v in grad_and_vars]
        vv = [v for g, v in grad_and_vars]
        scaled_val = val
        if do_summaries:
            clipped_tensors, g_norm, so = clip_by_global_norm_summary(
                grad_tensors, scaled_val, name, vv)
        else:
            so = []
            clipped_tensors, g_norm = tf.clip_by_global_norm(
                grad_tensors, scaled_val)

        ret = []
        for t, (g, v) in zip(clipped_tensors, grad_and_vars):
            ret.append((t, v))

        return ret, so

    all_clip_norm_val = options['all_clip_norm_val']
    ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

    assert len(ret) == len(grads)

    return ret, summary_ops


def train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=None, args=None):
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    import logging
    logger = logging.getLogger("lm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(str(args))
    logger.info(str(options))
    # not restarting so save the options
    if restart_ckpt_file is None and tf_save_dir:
        with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
            fout.write(json.dumps(options))

    with tf.device('/gpu:0'):
	with tf.device('/gpu:0'):
	    global_step = tf.get_variable(
		'global_step', [],
		initializer=tf.constant_initializer(0), trainable=False)
        # set up the optimizer
        lr = args.learning_rate
        if args.optim == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif args.optim == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=args.learning_rate)
        elif args.optim == 'rprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=args.learning_rate)
        else:
            opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        for k in range(1):
            with tf.device('/gpu:0'):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = LanguageModel(options, True, logger)
                    loss = model.total_loss
                    models.append(model)
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )

                    # keep track of loss across all GPUs
                    train_perplexity += loss

        # calculate the mean of each gradient across all GPUs
        grads2, norm_summary_ops = clip_grads(grads, options, True, global_step)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads2)

        init = tf.initialize_all_variables()

    # do the training loop
    bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            logger.info('load from checkpoint {}'.format(restart_ckpt_file))
            loader.restore(sess, restart_ckpt_file)
            
        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        print_debug_info(sess, logger, args=args)
        if tf_save_dir:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
            checkpoint_path = os.path.join(tf_save_dir, 'model_test.ckpt')
            logger.info('save to checkpoint {}'.format(checkpoint_path))
            saver.save(sess, checkpoint_path, global_step=0)

        save_para(sess, logger, args)

        logger.info("Training for %s epochs and %s batches" % (
            options['n_epochs'], n_batches_total))

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        fetch_vars = []
        grad_vars = []
        i = 0
        for model in models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)
            fetch_vars.append(model.token_ids)
            fetch_vars.append(model.token_ids_reverse)
            fetch_vars.extend(model.lstm_inputs)
            fetch_vars.extend(model.lstm_unpack)
            fetch_vars.extend(model.lstm_outputs)
            fetch_vars.extend(model.individual_losses)
            grad_vars.extend(model.lstm_inputs)
            grad_vars.extend(model.lstm_unpack)
            grad_vars.extend(model.lstm_outputs)
            grad_vars.extend(model.losses)
            grad_vars.extend(model.individual_losses)
            para = tf.trainable_variables()
            grad_vars = tf.gradients(ys=model.total_loss * options['unroll_steps'], xs=grad_vars)
            tmp = []
            for g,v in grads2:
                if not(g is None):
                    tmp.append(g)
            grad_vars.extend(tmp)
           
            if args.optim == 'adagrad':
                opt_slot = [opt.get_slot(v, 'accumulator') for v in para]
                grad_vars.extend(opt_slot)
            grad_para = tf.gradients(ys=model.total_loss * options['unroll_steps'], xs=para)

            i = i + 1

        feed_dict = {
            model.token_ids:
                np.zeros([batch_size, unroll_steps], dtype=np.int64)
            for model in models
        }

        if bidirectional:
            feed_dict.update({
                model.token_ids_reverse:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            })

        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        t1 = time.time()
        data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps)
        for batch_no, batch in enumerate(data_gen, start=1):

            # slice the input in the batch for the feed_dict
            X = batch
            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}
            for k in range(n_gpus):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(
                    _get_feed_dict_from_X(X, start, end, model,
                                          bidirectional)
                )

            # This runs the train_op, summaries and the "final_state_tensors"
            #   which just returns the tensors, passing in the initial
            #   state tensors, token ids and next token ids
            fetch_vars_len = len(fetch_vars)
            grad_vars_len = len(grad_vars)
            grad_para_len = len(grad_para)
            # also run the histogram summaries
            ret = sess.run(
                [train_op, train_perplexity] + fetch_vars + grad_vars + grad_para + 
                                            final_state_tensors,
                feed_dict=feed_dict
            )
            fetched_vars = ret[2:2 + fetch_vars_len]
            graded_vars = ret[2 + fetch_vars_len:2 + fetch_vars_len + grad_vars_len]
            graded_para = ret[2 + fetch_vars_len + grad_vars_len:2 + fetch_vars_len + grad_vars_len + grad_para_len]
            init_state_values = ret[2 + fetch_vars_len + grad_vars_len + grad_para_len:]
            def flatten(a):
                res = [y for x in a for y in x]
                return res

            k = final_state_tensors
            v = init_state_values
            k = flatten(flatten(k))
            v = flatten(flatten(v))
            if batch_no % args.log_interval == 0:
                print_debug_info(sess, logger, vars_data=(fetch_vars + k, fetched_vars + v), grad_data=(grad_vars, graded_vars), grad_para_data=(para, graded_para), args=args)
                # write the summaries to tensorboard and display perplexity
                logger.info("Batch %s, train ppl %s" % (batch_no, ret[1]))
                logger.info("Total time: %s" % (time.time() - t1))


            if batch_no > 100 and args.para_print:
                exit(0)
            if batch_no > 100 and args.detail:
                exit(0)

            if batch_no == n_batches_total:
                # done training!
                break

def test(options, ckpt_file, data, batch_size=256):
    '''
    Get the test set perplexity!
    '''

    bidirectional = options.get('bidirectional', False)

    unroll_steps = 1

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'), tf.variable_scope('lm'):
            test_options = dict(options)
            # NOTE: the number of tokens we skip in the last incomplete
            # batch is bounded above batch_size * unroll_steps
            test_options['batch_size'] = batch_size
            test_options['unroll_steps'] = 1
            model = LanguageModel(test_options, False, logger)
            # we use the "Saver" class to load the variables
            loader = tf.train.Saver()
            loader.restore(sess, ckpt_file)

        # model.total_loss is the op to compute the loss
        # perplexity is exp(loss)
        init_state_tensors = model.init_lstm_state
        final_state_tensors = model.final_lstm_state
        feed_dict = {
            model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
        }
        if bidirectional:
            feed_dict.update({
                model.token_ids_reverse:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
            })  

        init_state_values = sess.run(
            init_state_tensors,
            feed_dict=feed_dict)

        t1 = time.time()
        batch_losses = []
        total_loss = 0.0
        for batch_no, batch in enumerate(
                                data.iter_batches(batch_size, 1), start=1):
            # slice the input in the batch for the feed_dict
            X = batch

            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}

            feed_dict.update(
                _get_feed_dict_from_X(X, 0, X['token_ids'].shape[0], model, 
                                          bidirectional)
            )

            ret = sess.run(
                [model.total_loss, final_state_tensors],
                feed_dict=feed_dict
            )

            loss, init_state_values = ret
            batch_losses.append(loss)
            batch_perplexity = np.exp(loss)
            total_loss += loss
            avg_perplexity = np.exp(total_loss / batch_no)

            logger.info("batch=%s, batch_perplexity=%s, avg_perplexity=%s, time=%s" %
                (batch_no, batch_perplexity, avg_perplexity, time.time() - t1))

    avg_loss = np.mean(batch_losses)
    logger.info("FINSIHED!  AVERAGE PERPLEXITY = %s" % np.exp(avg_loss))

    return np.exp(avg_loss)

