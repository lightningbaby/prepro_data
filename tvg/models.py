from tvg.layers import  *
from tvg.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        # _init_用来对Model的实例做属性初始化，参数self，即这个被实例的object，**kwargs 是字典类参数
        # 在创建实例的时候，该方法马上被调用
        # 创建的时候，参数self不用输入，只需要后面的参数
        # kwargs接受字典类参数，如果是*args,则接受tuple类参数
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
            # 判断kwarg是否是 name，logging中的一个
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
            # self._class_指的是Class Model，而self指的是Model的当前这个实例
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            print("layers--in def build in models.py---", self.layers)
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        ###加softmax

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()#test

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))
        print("Dense in models.py")

    def predict(self):
        # return tf.nn.softmax(self.outputs)
        after_softmax=tf.nn.softmax(self.outputs)
        print("after_softmax-in class MLP in models.py--",after_softmax)
        return after_softmax

class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs) #初始化属性logging、vars、layers,activatios,inputs,outputs,loss,accuracy,optimizeer,opt_op

        self.inputs = placeholders['features']
        self.input_dim = input_dim # features[2][1]=8172
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        # print("input_dim--in class GCN in models.py---", self.input_dim)
        # 输入是4733项
        # print("output_dim--in class GCN in models.py---", self.output_dim)  # 7
        print("placeholders--in class GCN in models.py---", self.placeholders)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # learning rate设置为0.01
        # print("learning_rate in class GCN in models.py---", FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            # 利用 L2 范数来计算张量的误差值
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        # 在masked_accuracy函数里会比较self.outputs和self.placeholders['labels']
        # so 前者是分类器分类得到的结果，后者是正确的分类结果
        # 各有7列，猜测行数是4377
        print("self.outputs---def accuracy in models.py-",self.outputs,"\nself.placeholders['labels']--",self.placeholders['labels'])
        # print("self.outputs length--def accuracy in models.py",self.outputs.length(),"\nself.outputs--",self.outputs,"\nself.placeholders['labels']--",self.placeholders['labels'])

    def _build(self): # _build 是私有方法
        print("layer1-input_dim={}",self.input_dim) #1433  8172
        print("layer1-output_dim={}", FLAGS.hidden1) #16  16
        print("layer1-placeholders={}",self.placeholders)#features,support,labels_mask,droput,labels,num_features_nonzero
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        print("layer2-input_dim={}",FLAGS.hidden1)  # 16  16
        print("layer2-output_dim={}", self.output_dim)  # 7   10
        print("layer2-placeholders={}", self.placeholders)
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))



    def predict(self):
        # return tf.nn.softmax(self.outputs)
        after_softmax = tf.nn.softmax(self.outputs)
        print("after_softmax--in Class GCN in models.py",after_softmax)
        return after_softmax