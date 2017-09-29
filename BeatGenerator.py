
def BeatSynth():
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    #INPUTs
    soundFile = None
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    Y = tf.placeholder("float")

    #lstms are cool
    def bidirectionalLSTM():
        splitAudioOnDataLenght(sound)

        # Training Parameters
        learning_rate = 0.001
        training_steps = 10000
        batch_size = 10
        display_step = 200

        # Network Parameters
        num_input = 1  # MNIST data input (img shape: 28*28
        timesteps = 20000  # timesteps
        num_hidden = 128  # hidden layer num of features
        num_classes = 56  # MNIST total classes (0-56 digits)

        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])
        Y = tf.placeholder("float", [None, num_classes])

        # Define weights
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        def BiRNN(x, weights, biases):

            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
            x = tf.unstack(x, timesteps, 1)

            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            # Backward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

            # Get lstm cell output
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                             dtype=tf.float32)
            except Exception:  # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                       dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        logits = BiRNN(X, weights, biases)
        prediction = tf.nn.softmax(logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for step in range(1, training_steps + 1):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Reshape data to get 28 seq of 28 elements
                batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                         Y: batch_y})
                    print("Step " + str(step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

            print("Optimization Finished!")

            # Calculate accuracy for 128 mnist test images
            test_len = 128
            test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
            test_label = mnist.test.labels[:test_len]
            print("Testing Accuracy:", \
                  sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    def AutoEncoder():
        import tensorflow.contrib.rnn
        '''
        Build a 2 layers auto-encoder with TensorFlow to compress images to a
        lower latent space and then reconstruct them.

        USES MNIST Handwriting digits
        '''

        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        # Training Parameters
        learning_rate = 0.01
        num_steps = 30000
        batch_size = 256

        display_step = 1000
        examples_to_show = 10



        # Network Parameters
        num_input = 1  # MNIST data input (img shape: 1*bug)
        timesteps = 20000  # timesteps
        num_out = 1
        num_hidden = 128  # hidden layer num of features
        num_classes = 56  # MNIST total classes (0-56 digits)

        # tf Graph input
        X = tf.placeholder("float", [None, timesteps, num_input])       #So one rap beat will be broken down into parts of 2000, of chain length 1, and None Will be the batches of the rap beat. It should train on one rap beat at a time.
        Y = tf.placeholder("float", [None, num_classes])

        # Define weights
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'encoder_out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
            'decoder_h1': tf.Variable(tf.random_normal([num_classes, num_hidden])),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden, num_input])),
        }
        biases = {
            'encoder_out': tf.Variable(tf.random_normal([num_classes]))
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([num_input])),
        }

        # Network Parameters
        num_hidden_1 = 256  # 1st layer num features
        num_hidden_2 = 128  # 2nd layer num features (the latent dim)
        num_input = 784  # MNIST data input (img shape: 28*28)

        # tf Graph input (only pictures)
        X = tf.placeholder("float", [None, num_input])

        '''
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([num_classes, num_hidden])),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden, num_input])),

        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([num_input])),
        }
        '''

        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            # Encoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            return layer_2

        def BiRNNencoder(x, weights, biases):

            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, timesteps, n_input)
            # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

            # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
            x = tf.unstack(x, timesteps, 1)

            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
            # Backward direction cell
            lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

            # Get lstm cell output
            try:
                outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                             dtype=tf.float32)
            except Exception:  # Old TensorFlow version only returns outputs not states
                outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                       dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['encoder_out']) + biases['encoder_out']   #-> So this Bidirectional RNN produces num_classes

        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                           biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                           biases['decoder_b2']))
            return layer_2

        # Construct model

        encoder_op = BiRNNencoder(X, weights, biases)
        decoder_op = decoder(encoder_op)

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data because it IS AN IDENTITY FUNCTION
        y_true = X

        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start Training
        # Start a new TF session
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Training
            for i in range(1, num_steps + 1):
                # Prepare Data
                # Get the next batch of MNIST data (only images are needed, not labels)
                batch_x, _ = mnist.train.next_batch(batch_size)

                # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
                # Display logs per step
                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))

            # Testing
            # Encode and decode images from test set and visualize their reconstruction.
            n = 4
            canvas_orig = np.empty((28 * n, 28 * n))
            canvas_recon = np.empty((28 * n, 28 * n))
            for i in range(n):
                # MNIST test set
                batch_x, _ = mnist.test.next_batch(n)
                # Encode and decode the digit image
                g = sess.run(decoder_op, feed_dict={X: batch_x})

                # Display original images
                for j in range(n):
                    # Draw the original digits
                    canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                        batch_x[j].reshape([28, 28])
                # Display reconstructed images
                for j in range(n):
                    # Draw the reconstructed digits
                    canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                        g[j].reshape([28, 28])

            print("Original Images")
            plt.figure(figsize=(n, n))
            plt.imshow(canvas_orig, origin="upper", cmap="gray")
            plt.show()

            print("Reconstructed Images")
            plt.figure(figsize=(n, n))
            plt.imshow(canvas_recon, origin="upper", cmap="gray")
            plt.show()

    def GAN():
        def generator(x):
            w = tf.Variable(0.0, name="w1")

        def discriminator():
            2+2







