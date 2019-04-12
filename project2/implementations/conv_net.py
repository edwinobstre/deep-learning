"""
Implementation of convolutional neural network. Please implement your own convolutional neural network 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class ConvNet(object):
  """
  A convolutional neural network. 
  """

  def __init__(self, input_size, output_size, filter_size, pooling_schedule, fc_hidden_size,  weight_scale=None, centering_data=False, use_dropout=False, use_bn=False):
    """
    A suggested interface. You can choose to use a different interface and make changes to the notebook.

    Model initialization.

    Inputs:
    - input_size: The dimension D of the input data.
    - output_size: The number of classes C.
    - filter_size: sizes of convolutional filters
    - pooling_schedule: positions of pooling layers 
    - fc_hidden_size: sizes of hidden layers of hidden layers 
    - weight_scale: the initialization scale of weights
    - centering_data: Whether centering the data or not
    - use_dropout: whether use dropout layers. Dropout rates will be specified in training
    - use_bn: whether to use batch normalization

    Return: 
    """
    N, D, P = input_size
    C = output_size
    self.pooling_schedule = pooling_schedule
    
    tf.reset_default_graph()

    # record all options
    self.options = {'centering_data':centering_data, 'use_dropout':use_dropout, 'use_bn':use_bn}

    num_layers = len(filter_size)

    # construct the computational graph 
    #self.tf_graph = tf.Graph()
    #with self.tf_graph.as_default():
    # allocate parameters
    self.params = {'cW':[], 'cb':[], 'W':[], 'b':[]}
    
    # the first cnn layer
    size = 1
    
    if weight_scale is None:
           weight_scale = np.sqrt(2 / (N*D*P))
    if 0 in self.pooling_schedule:
           size = size * 4

    cW = tf.Variable(weight_scale * np.random.randn(filter_size[0][0], filter_size[0][1],
                                                                P, filter_size[0][2]),
                     dtype=tf.float32)
    cb = tf.Variable(0.01 * np.ones(filter_size[0][2]), dtype=tf.float32)

    self.params['cW'].append(cW)
    self.params['cb'].append(cb)
    
    # the rest cnn layers
    for ilayer in range(1,num_layers): 
        # the scale of the initialization
        if weight_scale is None:
            weight_scale = np.sqrt(2 / N * D * (1/size * filter_size[ilayer-1][2]))
        if ilayer in self.pooling_schedule:
            size = size * 4

        cW = tf.Variable(weight_scale * np.random.randn(filter_size[ilayer][0], filter_size[ilayer][1], filter_size[ilayer -1][2], filter_size[ilayer][2]), dtype=tf.float32)
        cb = tf.Variable(0.01 * np.ones(filter_size[ilayer][2]), dtype=tf.float32)

        self.params['cW'].append(cW)
        self.params['cb'].append(cb)
        
    # first linear layer
    previous_size = (int)(N * D * filter_size[-1][2] / size)
    if weight_scale is None:
        weight_scale = np.sqrt(2 / previous_size)
        
    W = tf.Variable(weight_scale * np.random.randn(previous_size, fc_hidden_size[0]), dtype=tf.float32)
    b = tf.Variable(0.01 * np.ones(fc_hidden_size[0]), dtype=tf.float32)

    self.params['W'].append(W)
    self.params['b'].append(b)
    
    #last linear layer
        
    if weight_scale is None:
        weight_scale = np.sqrt(2 / fc_hidden_size[0])

    W = tf.Variable(weight_scale * np.random.randn(fc_hidden_size[0], output_size), dtype=tf.float32)
    b = tf.Variable(0.01 * np.ones(output_size), dtype=tf.float32)

    self.params['W'].append(W)
    self.params['b'].append(b)


    # allocate place holders 
    
    self.placeholders = {}

    # data feeder
    self.placeholders['x_batch'] = tf.placeholder(dtype=tf.float32, shape=[None, N,D,P])
    self.placeholders['y_batch']= tf.placeholder(dtype=tf.int32, shape=[None])

    # the working mode 
    self.placeholders['training_mode'] = tf.placeholder(dtype=tf.bool, shape=())
    
    # data center 
    self.placeholders['x_center'] = tf.placeholder(dtype=tf.float32, shape=[N,D,P])

    # keeping probability of the droput layer
    self.placeholders['keep_prob'] = tf.placeholder(dtype=tf.float32, shape=[])

    # regularization weight 
    self.placeholders['reg_weight'] = tf.placeholder(dtype=tf.float32, shape=[])


    # learning rate
    self.placeholders['learning_rate'] = tf.placeholder(dtype=tf.float32, shape=[])
    
    self.operations = {}

    # construct graph for score calculation 
    scores = self.compute_scores(self.placeholders['x_batch'])
                            
    # predict operation
    self.operations['y_pred'] = tf.argmax(scores, axis=-1)


    # construct graph for training 
    objective = self.compute_objective(scores, self.placeholders['y_batch'])
    self.operations['objective'] = objective

    minimizer = tf.train.AdamOptimizer(learning_rate=self.placeholders['learning_rate'])
    training_step = minimizer.minimize(objective)

    self.operations['training_step'] = training_step 

    if self.options['use_bn']:
        bn_update = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    else: 
        bn_update = []
    self.operations['bn_update'] = bn_update


    # maintain a session for the entire model
    self.session = tf.Session()

    self.x_center = None # will get data center at training

    return 

  def softmax_loss(self, scores, y):
        """
        Compute the softmax loss. Implement this function in tensorflow

        Inputs:
        - scores: Input data of shape (N, C), tf tensor. Each scores[i] is a vector 
          containing the scores of instance i for C classes .
        - y: Vector of training labels, tf tensor. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength, scalar.

        Returns:
        - loss: softmax loss for this batch of training samples.
        """
    
        # 
        # Compute the loss

        softmax_loss =tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=scores))

        return softmax_loss
    
  def regularizer(self):
        """ 
        Calculate the regularization term
        Input: 
        Return: 
            the regularization term
        """
        reg = np.float32(0.0)
        for cW in self.params['cW']:
            reg = reg + self.placeholders['reg_weight'] * tf.reduce_sum(tf.square(cW))
        for W in self.params['W']:
            reg = reg + self.placeholders['reg_weight'] * tf.reduce_sum(tf.square(W))
    
        return reg
    
    
  def compute_scores(self, X):
        """

        Compute the loss and gradients for a two layer fully connected neural
        network. Implement this function in tensorflow

        Inputs:
        - X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.
    
        Returns:
        - scores: a tensor of shape (N, C) where scores[i, c] is the score for 
          class c on input X[i].

        """
    
        # Unpack variables from the params dictionary
    
        if self.options['centering_data']:  
            X = X - self.placeholders['x_center']

        num_layers = len(self.params['cW']) + len(self.params['W'])

        hidden = X
 
        for ilayer in range(0, num_layers):
            # the last layer linear transform
            if ilayer == (num_layers - 1):  
                W = self.params['W'][1]
                b = self.params['b'][1]
                hidden = tf.matmul(hidden, W) + b

            # otherwise optionally apply batch normalization, relu, and dropout to all layers 
            else:
                if ilayer == (num_layers - 2):
                    W = self.params['W'][0]
                    b = self.params['b'][0]
                    hidden_size = hidden.get_shape().as_list()
                    hidden = tf.reshape(hidden, [-1, hidden_size[1]*hidden_size[2]*hidden_size[3]])
                    conv_trans = tf.matmul(hidden, W) + b
                else:
                    W = self.params['cW'][ilayer]
                    b = self.params['cb'][ilayer]
                    conv_trans = tf.nn.conv2d(hidden, W, strides=[1,1,1,1], padding='SAME') + b
                    
                # use dropout
                if self.options['use_dropout']:
                    conv_trans = tf.cond(self.placeholders['training_mode'],
                                         lambda:tf.nn.dropout(conv_trans, 
                                                              keep_prob=self.placeholders['keep_prob']))
            
                # use batch normalization
                if self.options['use_bn']:
                    conv_trans = tf.layers.batch_normalization(conv_trans, 
                                         training=self.placeholders['training_mode'])
                
                # non-linear transformation
                hidden = tf.nn.relu(conv_trans)
                
                # max pooling
                if ilayer in self.pooling_schedule:
                    hidden = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    
        scores = hidden

        return scores
    
  def compute_objective(self, scores, y):
        """
        Compute the training objective of the neural network.

        Inputs:
        - scores: A numpy array of shape (N, C). C scores for each instance. C is the number of classes 
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - reg: a np.float32 scalar

        Returns: 
        - objective: a tensorflow scalar. the training objective, which is the sum of 
          losses and the regularization term
        """

        # get output size, which is the number of classes
        num_classes = self.params['b'][-1].get_shape()[0]

        y1hot = tf.one_hot(y, depth=num_classes)
        loss = self.softmax_loss(scores, y1hot)

        reg_term = self.regularizer()

        objective = loss + reg_term

        return objective

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=1.0, keep_prob=1.0, 
            reg=np.float32(5e-6), num_iters=100,
            batch_size=200, verbose=False):
    """
    A suggested interface of training the model.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - keep_prob: probability of keeping values when using dropout
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)
    num_classes = self.params['b'][-1].get_shape()[0]
    num_layers = len(self.params['cW']) + len(self.params['W'])

    self.x_center = np.mean(X, axis=0)

    ############################################################################
    # after this line, you should execute appropriate operations in the graph to train the mode  

    session = self.session
    session.run(tf.global_variables_initializer())

    # Use SGD to optimize the parameters in self.model
    objective_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):

        b0 = (it * batch_size) % num_train 
        batch = range(b0, min(b0 + batch_size, num_train))

        X_batch = X[batch]
        y_batch = y[batch] 

        feed_dict = {self.placeholders['x_batch']: X_batch, 
                         self.placeholders['y_batch']: y_batch, 
                         self.placeholders['learning_rate']:learning_rate, 
                         self.placeholders['training_mode']:True, 
                         self.placeholders['reg_weight']:reg}

        # Decay learning rate
        learning_rate *= learning_rate_decay


        if self.options['centering_data']:
            feed_dict[self.placeholders['x_center']] = self.x_center

        if self.options['use_dropout']:
            feed_dict[self.placeholders['keep_prob']] = np.float32(keep_prob)

        if self.options['use_bn']:
            np_objective, _, _= session.run([self.operations['objective'],
                                          self.operations['training_step'],
                                          self.operations['bn_update']], feed_dict=feed_dict)
        else:
            np_objective, _= session.run([self.operations['objective'],
                                          self.operations['training_step']
                                          ], feed_dict=feed_dict)

        objective_history.append(np_objective)

        if verbose and it % 100 == 0:
            print('iteration %d / %d: objective %f' % (it, num_iters, np_objective))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            train_acc = np.float32(self.predict(X_batch) == y_batch).mean()
            val_acc = np.float32(self.predict(X_val) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
    
    return {
          'objective_history': objective_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
    }
    
    

  def predict(self, X):
    """
    Use the trained weights of the neural network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    np_y_pred = self.session.run(self.operations['y_pred'], 
                                     feed_dict={self.placeholders['x_batch']: X,                                                                       
                                                self.placeholders['training_mode']:False, 
                                                                           
                                                self.placeholders['x_center']:self.x_center, 
                                                                           
                                                self.placeholders['keep_prob']:1.0})
                                                                           
        
    return np_y_pred

  def get_params(self):
      params = {'filter':[]}
      params['filter'] = self.params['cW'][0].eval(session=self.session)
        
      return params


