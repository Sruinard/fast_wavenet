class Generator(object):
    def __init__(self, model, batch_size=1, input_size_engine=6, input_size_condition=4):
        self.model = model
        self.bins = load_data('/Users/stefruinard/Desktop/FastWavenet/fitters/', 'created_bins')

        inputs_engine = tf.placeholder(tf.float32, [batch_size, input_size_engine],
                                       name='inputs_engine')
        inputs_condition = tf.placeholder(tf.float32, [batch_size, input_size_condition],
                                          name='inputs_condition')

        print('Make Generator.')

        count = 0
        h_engine = inputs_engine
        h_condition = inputs_condition

        init_ops = []
        push_ops = []
        for b in range(self.model.n_blocks):
            for i in range(self.model.n_layers):
                rate = 2 ** i
                name = 'b{}-l{}'.format(b, i)
                if count == 0:

                    # create queues and initialize
                    q_inputs = tf.FIFOQueue(rate,
                                            dtypes=tf.float32,
                                            shapes=(batch_size, input_size_engine))
                    init_engine = q_inputs.enqueue_many(tf.zeros((rate, batch_size, input_size_engine)))

                    q_condition = tf.FIFOQueue(rate,
                                               dtypes=tf.float32,
                                               shapes=(batch_size, input_size_condition))
                    init_condition = q_condition.enqueue_many(tf.zeros((rate, batch_size, input_size_condition)))

                    # get state and put new observation in queue
                    state_engine_ = q_inputs.dequeue()
                    push_engine = q_inputs.enqueue([h_engine])
                    state_condition_ = q_condition.dequeue()
                    push_condition = q_condition.enqueue([h_engine])

                    # add ops
                    init_ops.append(init_engine)
                    push_ops.append(push_engine)
                    init_ops.append(init_condition)
                    push_ops.append(push_condition)

                    h = _causal_linear(h_engine, h_condition, state_engine_, state_condition_, name=name,
                                       activation=tf.nn.relu)
                    count += 1

                else:
                    state_size = self.model.n_channels_per_layer

                    q = tf.FIFOQueue(rate,
                                     dtypes=tf.float32,
                                     shapes=(batch_size, state_size))
                    init = q.enqueue_many(tf.zeros((rate, batch_size, state_size)))

                    state_ = q.dequeue()
                    push = q.enqueue([h])
                    init_ops.append(init)
                    push_ops.append(push)

                    h = _causal_linear(h, state_, name=name, activation=tf.nn.relu)
                    count += 1

        outputs = _output_linear(h)

        out_ops = [tf.nn.softmax(outputs)]
        out_ops.extend(push_ops)

        self.inputs = inputs
        self.init_ops = init_ops
        self.out_ops = out_ops

        # Initialize queues.
        self.model.sess.run(self.init_ops)

    def run(self, input, num_samples):
        predictions = []
        for step in range(num_samples):

            feed_dict = {self.inputs: input}
            output = self.model.sess.run(self.out_ops, feed_dict=feed_dict)[0]  # ignore push ops
            value = np.argmax(output[0, :])

            input = np.array(self.bins[value])[None, None]
            predictions.append(input)

            if step % 1000 == 0:
                predictions_ = np.concatenate(predictions, axis=1)
                plt.plot(predictions_[0, :], label='pred')
                plt.legend()
                plt.xlabel('samples from start')
                plt.ylabel('signal')
                plt.show()

        predictions_ = np.concatenate(predictions, axis=1)
        return predictions_