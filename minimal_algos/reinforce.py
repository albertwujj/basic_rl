import random
import tensorflow as tf
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
from spinup.utils.logx import EpochLogger
import time
from env_wrappers.frozenlake import EnvWrapper
from nn_utils.mlp import mlp
from test_utils.eval_model import eval_model

class Model():

    def __init__(self, act_dim, obs_dim):
        obs_g = tf.placeholder(tf.float32, (None, obs_dim))
        a_g = tf.placeholder(tf.int32, (None, 1))
        returns_g = tf.placeholder(tf.float32, (None, 1))

        logits_g = mlp(obs_g, [5, 5, act_dim], tf.nn.relu, None)
        log_prob_a_g = tf.nn.log_softmax(logits_g)
        prob_a_g = tf.nn.softmax(logits_g)
        one_hot_a = tf.one_hot(tf.squeeze(a_g, axis=-1), act_dim, axis=-1)
        selected_log_prob_a_g = tf.boolean_mask(log_prob_a_g, one_hot_a)

        loss_g = tf.reduce_mean(-selected_log_prob_a_g * returns_g)
        train_step = tf.train.AdamOptimizer().minimize(loss_g)

        sample_a = tf.multinomial(logits_g, 1)
        max_a = tf.argmax(log_prob_a_g, 1)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        def train(obs, a, returns):
            a = np.expand_dims(a, -1)
            returns = np.expand_dims(returns, -1)
            _, loss = sess.run([train_step, loss_g], feed_dict = {obs_g: obs, a_g:a, returns_g:returns})
            return loss

        def sample_act(obs):
            return sess.run([sample_a], feed_dict={obs_g: obs})[0]

        def choose_act(obs):
            return sess.run([max_a, prob_a_g], feed_dict={obs_g: obs})

        self.train = train
        self.sample_act = sample_act
        self.choose_act = choose_act

def scalarize(arr):
    return np.argmax(arr, -1)

def main():
    env = EnvWrapper(FrozenLakeEnv(is_slippery=False, map_name='4x4'))
    model = Model(env.action_space.n, env.observation_space.n)
    logger = EpochLogger()

    start_time = time.time()
    gamma = 1

    for epoch in range(5000000):

        done = False
        obs_arr, rew_arr, val_arr, act_arr = [], [], [], []
        obs_s_arr = []
        total_rew = 0

        obs = env.reset()
        while not done:
            obs_arr.append(obs)
            obs_s_arr.append(scalarize(obs))

            a_t = model.sample_act([obs])
            a = a_t[0][0] # array of dim 1
            act_arr.append(a)
            obs, rew, done, _ = env.step(a)
            rew_arr.append(rew)
            total_rew += rew

        val = 0
        for i in range(len(rew_arr) - 1, -1, -1):
            val *= gamma
            val += rew_arr[i]
            val_arr.append(val)
        val_arr.reverse()

        loss = model.train(obs_arr, act_arr, val_arr)
        logger.store(Loss = loss, Return = total_rew)
        if (epoch + 1) % 10000 == 0:
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('Return', average_only=True)
            logger.log_tabular('Loss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

            env.viz_actions(model)

    totalRet = 0
    eps = 1000

    print(eval_model(model, env))

main()