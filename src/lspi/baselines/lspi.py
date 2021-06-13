from collections import namedtuple

import numpy as np

Sample = namedtuple('Sample', ['s', 'a', 'r', 's_'])


class LSPolicyIteration:
    def __init__(self,
                 env,
                 agent,
                 gamma,
                 memory_size,
                 memory_type='sample',
                 eval_type='batch'):
        """Least-Squares Policy Iteration algorithm

        Args:
            env (gym.Env): gym environment.
            agent (lspi.agents.Agent): features policy.
            gamma (float): discount factor.
            memory_size (int): number of training samples/episodes.  samples/episodes分别是啥意思
            memory_type (str, optional): samples collecting method. Defaults to 'sample'.
            eval_type (str, optional): policy evaluation method. Defaults to 'batch'.
        """
        if not memory_type in ['sample', 'episode']:
            raise ValueError(
                "memory_type can take values ['sample','episode']")
        if not eval_type in ['iterative', 'sherman_morrison', 'batch']:
            raise ValueError(
                "eval_type can take values ['iterative','sherman_morrison','batch']"
            )
        self.env = env
        self.gamma = gamma
        self.agent = agent
        self.memory_size = memory_size
        self.eval_type = eval_type
        self.memory_type = memory_type

    def init_memory(self):
        """
        如果是episode类型的将采样得到的数据存储在memory中
        """
        self.memory = []
        count = 0
        done = True
        while count < (self.memory_size + 1):
            if done:
                obs = self.env.reset()
                if self.memory_type == 'episode': #samples/episodes分别是啥意思
                    count += 1
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            self.memory.append(Sample(obs, action, reward, next_obs))
            obs = next_obs
            if self.memory_type == 'sample':#samples/episodes分别是啥意思
                count += 1

        if self.eval_type == 'batch':
            k = self.agent.features_size
            nActions = self.agent.action_size
            self.A_all = np.zeros(
                (len(self.memory), nActions, k * nActions, k * nActions))#4维啥意思
            self.b_all = np.zeros(k * nActions)
            """
            怎么知道A，b数组的维数的
            """
            for idx, sample in enumerate(self.memory):
                # state features
                feat_s = np.zeros(k * nActions)#开辟一个新的数组
                #为啥特征的维数要乘以nActions
                a = sample.a #动作
                feat_s[a * k:(a + 1) * k] = self.agent.get_features(sample.s) 
                #状态为s的特征存储在feat_s[a * k:(a + 1) * k]里，为啥是a * k:(a + 1) * k呢？
                #因为a是在s下选择的，又因为每个特征要占k个索引区间（k = self.agent.features_size）（猜的）
                #[m:n] #切片操作，取a[m]~a[n-1]之间的内容，m\n可以为负，m>n时返回空
                
                # next state features
                feat_ = self.agent.get_features(sample.s_)
                for a_ in range(nActions):
                    feat_s_ = np.zeros(k * nActions)
                    feat_s_[a_ * k:(a_ + 1) * k] = feat_
                    self.A_all[idx, a_, :, :] = np.outer(
                        feat_s, feat_s - self.gamma * feat_s_)
                # reward features
                self.b_all += sample.r * feat_s

    def eval(self):
        k = self.agent.features_size
        nActions = self.agent.action_size
        if self.eval_type == 'iterative':
            A = np.zeros((k * nActions, k * nActions))
            b = np.zeros(k * nActions)
            for sample in self.memory:
                # state features
                feat_s = np.zeros(k * nActions)
                a = sample.a
                feat_s[a * k:(a + 1) * k] = self.agent.get_features(sample.s)
                # next state features
                feat_s_ = np.zeros(k * nActions)
                a_ = self.agent.predict(sample.s_)
                feat_s_[a_ * k:(a_ + 1) * k] = self.agent.get_features(
                    sample.s_)
                # update parameters
                A += np.outer(feat_s, feat_s - self.gamma * feat_s_)
                b += sample.r * feat_s
            w = np.linalg.solve(A, b)
        elif self.eval_type == 'sherman_morrison':
            B = np.eye(k * nActions)
            b = np.zeros(k * nActions)
            for sample in self.memory:
                # state features
                feat_s = np.zeros(k * nActions)
                a = sample.a
                feat_s[a * k:(a + 1) * k] = self.agent.get_features(sample.s)
                # next state features
                feat_s_ = np.zeros(k * nActions)
                a_ = self.agent.predict(sample.s_)
                feat_s_[a_ * k:(a_ + 1) * k] = self.agent.get_features(
                    sample.s_)
                # update matrix
                B -= np.outer(np.dot(
                    B, feat_s), np.dot(
                        B.T, feat_s - self.gamma * feat_s_)) / (1 + np.inner(
                            feat_s - self.gamma * feat_s_, np.dot(B, feat_s)))
                b += sample.r * feat_s
            w = np.dot(B, b)
        elif self.eval_type == 'batch':
            A = np.array([
                self.A_all[idx, self.agent.predict(sample.s_)]
                for idx, sample in enumerate(self.memory)
            ]).sum(0)
            #sum(0) 用途：求数组每一列的和，等价于sum(axis=0),   1 行和
            b = self.b_all
            w = np.linalg.solve(A, b)
        return w

    def train_step(self):
        w = self.eval()
        self.agent.set_weights(w)
