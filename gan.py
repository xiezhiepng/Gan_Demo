import argparse
import numpy as np
from scipy.stats import norm#统计分析相关库：高斯分布
import tensorflow as tf
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation 
import seaborn as sns#可视化库

sns.set(color_codes=True)  

seed = 42
np.random.seed(seed) 
tf.set_random_seed(seed)


class DataDistribution(object):
    def __init__(self):#高斯分布两个参数
        self.mu = 4#均值
        self.sigma = 0.5#标注差

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples


class GeneratorDistribution(object):#制造噪声，随机初始化分布
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01


def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)#定义随机初始化
    const = tf.constant_initializer(0.0)#
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b#return input*w+b


def generator(input, h_dim):#输入 12*1的shape
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))# 初始化w参数，初始化b参数
    h1 = linear(h0, 1, 'g1')
    return h1#得到h1 最终生成结果   只有两层


def discriminator(input, h_dim):#h_dim神经元个数
    #网络分层
    h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))#linear函数控制w,b大小   h_dim神经元个数
    h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))   #神经元个数 12*8
    h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))#sigmoid激活函数 最后的输出层得到预测结果
    return h3

def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95#学习率不断衰减学习策略 ： 定义衰减的策略
    num_decay_steps = 150#迭代150次进行学习率衰减
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    #定义求解器 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, log_every):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.log_every = log_every
        self.mlp_hidden_size = 4#定义神经元个数
        
        self.learning_rate = 0.03#学习率

        self._create_model()

    def _create_model(self):#构建模型骨架

        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))#变量域：D网络 （12,1）shaoe
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))#定义初始化预测值
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size)#对w，b初始化，得到预测值
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))#定义loss(预测值与label之间差)
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)#定义优化器进行求解：学习率不断衰减的策略

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('Gen'):#G网络
            #噪声输入---》真实输入
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))#噪音输入
            self.G = generator(self.z, self.mlp_hidden_size)#最终输出结果 --- 一维点生成

        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('Disc') as scope: #D网络
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size)#构造D网络 输入：x是真实数据
            scope.reuse_variables()#变量重新使用，不需要重新定义
            self.D2 = discriminator(self.G, self.mlp_hidden_size)#D2是生成的数据

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        #定义损失函数
        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))#判别网络损失函数 log(self.D1)真实数据输入，希望D1=0
        #self.D2 生成数据--》趋进1      
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))#生成网络损失函数  希望D2=1  loss(D2)=0
        

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
        
        #优化器，求解器不断优化lossD,lossG

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        with tf.Session() as session:
            tf.global_variables_initializer().run()#变量全局初始化

            # pretraining discriminator #预训练网络D
            num_pretrain_steps = 1000
           
            for step in range(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0
                labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)#
            # copy weights from pre-training over to new D network
            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))#参数初始化，不断往真正D网络中来

            for step in range(self.num_steps):#训练真正GAN网络
                # update discriminator
                x = self.data.sample(self.batch_size)#真实数据
                z = self.gen.sample(self.batch_size)#噪音初始化，最后目标：通过G生成真实数据点
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                # update generator
                z = self.gen.sample(self.batch_size)#生成噪音，随机初始化
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })
    
                #不断迭代优化

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))#打印loss值                
                if step % 100 == 0 or step==0 or step == self.num_steps -1 :
                    self._plot_distributions(session)#绘图

    def _samples(self, session, num_points=10000, num_bins=100):
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return pd, pg

    def _plot_distributions(self, session):
        pd, pg = self._samples(session)
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()
def main(args):
    model = GAN(
        DataDistribution(),#定义真实数据分布
        GeneratorDistribution(range=8),#制造噪声，随机初始化分布
        args.num_steps,
        args.batch_size,#迭代1200次
        args.log_every,#间隔多少次打印当前loss
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=1200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
