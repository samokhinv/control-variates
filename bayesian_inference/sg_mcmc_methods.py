import torch
import copy
from torch.optim import Optimizer
from abc import ABC, abstractmethod

#from numpy.random import gamma
# class BaseOptimizer(Optimizer):
#     def __init__(self, params, defaults):
#         super().__init__(params, defaults)
#         self.alpha0 = 1
#         self.beta0 = 1

#     def step(self, burn_in=None, resample_momentum=None, resample_prior=None):
#         raise NotImplementedError

#     def resample_prior(self, p):
#         alpha = self.alpha0 + p.data.nelement() / 2
#         beta = self.beta0 + (p.data ** 2).sum().item() / 2
#         return gamma(shape=alpha, scale=1 / beta)  # gamma sample


class SG_MCMC(Optimizer):
    def init(self, params):
        self.__init__(params, **self.defaults)


class SGLD(SG_MCMC):
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 #weight_decay: float = 0.5,
                 #alpha0: float = 1,
                 #beta0: float = 1,
                 ):

        #self.alpha0 = alpha0
        #self.beta0 = beta0
        #self.weight_decay = gamma(shape=self.alpha0, scale=1/self.beta0) if sample_prior is True else 0.01

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        #if weight_decay <= 0.0:
        #    raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.defaults = dict(lr=lr)

        super().__init__(params, self.defaults)

    @torch.no_grad()
    def step(self, burn_in=False, resample_momentum=False):  # the same call as for HMC
        loss = None
        flat_grad = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                #if len(state) == 0:
                #    state['weight_decay'] = self.weight_decay
                #if resample_prior:
                #    state['weight_decay'] = self.resample_prior(p)

                d_p = p.grad
                #weight_decay = state['weight_decay']

                #if weight_decay != 0:
                #    d_p.add_(p, alpha=weight_decay)

                flat_grad.append(d_p.flatten())

                unit_noise = p.new_empty(p.size()).normal_()
                p.add_(0.5 * d_p + unit_noise / group['lr'] ** 0.5, alpha=-group['lr'])
        flat_grad = torch.cat(flat_grad, dim=0)
        return loss, flat_grad


class SVRG_LD(SG_MCMC):
    def __init__(self,
                 params,
                 lr: float = 1e-3,
                 sample_lr:float = None,
                 ):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        sample_lr = lr if sample_lr is None else sample_lr
        self.defaults = dict(lr=lr, sample_lr=sample_lr)

        super().__init__(params, self.defaults)

        self.dataset_grads = [{'params': [0. for _ in g['params']]} for g in self.param_groups]
        self.batch_grads = [{'params': [0. for _ in g['params']]} for g in self.param_groups]

    @torch.no_grad() 
    def load_dataset_grads(self, dataset_param_groups):
      for group, group_d in zip(self.dataset_grads, dataset_param_groups):
            for i, p_d in enumerate(group_d['params']):
                if p_d.grad is None:
                    continue
                group['params'][i] = p_d.grad.detach().clone()

    @torch.no_grad()
    def load_batch_grads(self, batch_param_groups):
      for group, group_b in zip(self.batch_grads, batch_param_groups):
            for i, p_b in enumerate(group_b['params']):
                if p_b.grad is None:
                    continue
                group['params'][i] = p_b.grad.detach().clone()

    @torch.no_grad()
    def get_grad(self):
        flat_grad = []
        for group, group_d, group_b in zip(self.param_groups, self.dataset_grads, self.batch_grads):
            for p, d_d, d_b in zip(group['params'], group_d['params'], group_b['params']):
                if p.grad is None:
                    continue

                d_p = p.grad
                d = d_d + (d_p - d_b)

                flat_grad.append(d.flatten())

        flat_grad = torch.cat(flat_grad, dim=0)
        return flat_grad

    @torch.no_grad()
    def step(self, burn_in=False, resample_momentum=False):  # the same call as for HMC
        loss = None
        flat_grad = []
        for group, group_d, group_b in zip(self.param_groups, self.dataset_grads, self.batch_grads):
            for p, d_d, d_b in zip(group['params'], group_d['params'], group_b['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                d = d_d + (d_p - d_b)
                flat_grad.append(d.flatten())

                unit_noise = p.new_empty(p.size()).normal_()
                if burn_in is True:
                    lr = group['lr']
                else:
                    lr = group['sample_lr']
                p.add_(d + 2**0.5 * unit_noise / lr ** 0.5, alpha=-lr)
        flat_grad = torch.cat(flat_grad, dim=0)
        return loss, flat_grad


class ScaleAdaSGHMC(SG_MCMC):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses scale adaption during burn-in
        procedure to find some hyperparamters. A gaussian prior is placed over parameters and a Gamma
        Hyperprior is placed over the prior's standard deviation

        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.

        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
        """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 base_c: float = 0.05,
                 sample_lr:float = None,
                #  gauss_sig: float = 0.1,
                #  alpha0: float = 1,
                #  beta0: float = 1,
                 ):
        """
        Set up the optimizer

        :param params: Iterable, parameters serving as optimization variables
        :param lr: float, learning
        :param base_c: float, friction term
        :param gauss_sig: float, initial prior sigma
        :param alpha0: float, initial hyperprior
        :param beta0: flaot, initial hyperprior
        """
        self.eps = 1e-6
        #self.alpha0 = alpha0
        #self.beta0 = beta0
        #self.weight_decay = gamma(shape=self.alpha0, scale=1/self.beta0) if sample_prior is True else 0.01

        #if self.weight_decay <= 0.0:
        #    raise ValueError("Invalid weight_decay value: {}".format(self.weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_c < 0:
            raise ValueError("Invalid friction term: {}".format(base_c))

        self.defaults = dict(
            lr=lr,
            base_c=base_c,
        )
        super().__init__(params, self.defaults)

    @torch.no_grad()
    def step(self, burn_in=False, resample_momentum=False):
        """Simulate discretized Hamiltonian dynamics for one step"""
        loss = None
        flat_grad = []
        for group in self.param_groups:  # iterate over blocks -> the ones defined in defaults. We dont use groups.
            for p in group["params"]:  # these are weight and bias matrices
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(
                        p)  # p.data.new(p.data.size()).normal_(mean=0, std=np.sqrt(group["lr"])) #
                    #state['weight_decay'] = self.weight_decay

                state["iteration"] += 1  # this is kind of useless now but lets keep it provisionally

                #if resample_prior:
                #    state['weight_decay'] = self.resample_prior(p)

                base_c, lr = group["base_c"], group["lr"]
                #weight_decay = state["weight_decay"]
                tau, g, v_hat = state["tau"], state["g"], state["V_hat"]

                d_p = p.grad
                #if weight_decay != 0:
                #    d_p.add_(p.data, alpha=weight_decay)
                flat_grad.append(d_p.flatten())
                # update parameters during burn-in
                if burn_in:  # We update g first as it makes most sense
                    tau.add_(-tau * (g ** 2) /
                             (v_hat + self.eps) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    tau_inv = 1. / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d_p)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-tau_inv * v_hat + tau_inv * (d_p ** 2))  # gradient variance see Eq 8 in [1]

                v_sqrt = torch.sqrt(v_hat)
                v_inv_sqrt = 1. / (v_sqrt + self.eps)  # preconditioner

                if resample_momentum:  # equivalent to var = M under momentum reparametrisation
                    state["v_momentum"] = torch.normal(mean=torch.zeros_like(d_p),
                                                       std=lr * v_inv_sqrt)
                v_momentum = state["v_momentum"]

                noise_var = (2. * (lr ** 3) * v_inv_sqrt * base_c * v_inv_sqrt - (lr ** 4))
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                # sample random epsilon
                noise_sample = torch.normal(mean=torch.zeros_like(d_p), std=noise_std)

                # update momentum (Eq 10 right in [1])
                v_momentum.add_(- (lr ** 2) * v_inv_sqrt * d_p - base_c * v_momentum + noise_sample)

                # update theta (Eq 10 left in [1])
                p.add_(v_momentum)
        flat_grad = torch.cat(flat_grad, dim=0)
        return loss, flat_grad


class SVRG_HMC(SG_MCMC):
    def __init__(self,
                 params,
                 lr: float = 1e-1,
                 base_c: float = 0.05,
                 sample_lr:float = None,
                 ):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_c < 0:
            raise ValueError("Invalid friction term: {}".format(base_c))

        sample_lr = lr if sample_lr is None else sample_lr
        self.eps = 1e-6
        
        self.defaults = dict(
            lr=lr,
            base_c=base_c,
            sample_lr=sample_lr
        )
        super().__init__(params, self.defaults)

        self.dataset_grads = [{'params': [0. for _ in g['params']]} for g in self.param_groups]
        self.batch_grads = [{'params': [0. for _ in g['params']]} for g in self.param_groups]

    @torch.no_grad() 
    def load_dataset_grads(self, dataset_param_groups):
      for group, group_d in zip(self.dataset_grads, dataset_param_groups):
            for i, p_d in enumerate(group_d['params']):
                if p_d.grad is None:
                    continue
                group['params'][i] = p_d.grad.detach().clone()

    @torch.no_grad()
    def load_batch_grads(self, batch_param_groups):
      for group, group_b in zip(self.batch_grads, batch_param_groups):
            for i, p_b in enumerate(group_b['params']):
                if p_b.grad is None:
                    continue
                group['params'][i] = p_b.grad.detach().clone()

    @torch.no_grad()
    def get_grad(self):
        flat_grad = []
        for group, group_d, group_b in zip(self.param_groups, self.dataset_grads, self.batch_grads):
            for p, d_d, d_b in zip(group['params'], group_d['params'], group_b['params']):
                if p.grad is None:
                    continue

                d_p = p.grad
                d = d_d + (d_p - d_b)

                flat_grad.append(d.flatten())

        flat_grad = torch.cat(flat_grad, dim=0)
        return flat_grad

    @torch.no_grad()
    def step(self, burn_in=False, resample_momentum=False):
        loss = None
        flat_grad = []
        for group, group_d, group_b in zip(self.param_groups, self.dataset_grads, self.batch_grads):
            for p, d_d, d_b in zip(group['params'], group_d['params'], group_b['params']):
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(
                        p)  # p.data.new(p.data.size()).normal_(mean=0, std=np.sqrt(group["lr"])) #
                    #state['weight_decay'] = self.weight_decay

                state["iteration"] += 1  # this is kind of useless now but lets keep it provisionally

                #if resample_prior:
                #    state['weight_decay'] = self.resample_prior(p)

                base_c, lr = group["base_c"], group["lr"]
                #weight_decay = state["weight_decay"]
                tau, g, v_hat = state["tau"], state["g"], state["V_hat"]

                d_p = p.grad
                #if weight_decay != 0:
                #    d_p.add_(p.data, alpha=weight_decay)
                d = d_d + (d_p - d_b)
                flat_grad.append(d.flatten())

                # update parameters during burn-in
                if burn_in:  # We update g first as it makes most sense
                    tau.add_(-tau * (g ** 2) /
                             (v_hat + self.eps) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    tau_inv = 1. / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-tau_inv * v_hat + tau_inv * (d ** 2))  # gradient variance see Eq 8 in [1]

                v_sqrt = torch.sqrt(v_hat)
                v_inv_sqrt = 1. / (v_sqrt + self.eps)  # preconditioner

                if resample_momentum:  # equivalent to var = M under momentum reparametrisation
                    state["v_momentum"] = torch.normal(mean=torch.zeros_like(d),
                                                       std=lr * v_inv_sqrt)
                v_momentum = state["v_momentum"]

                noise_var = (2. * (lr ** 3) * v_inv_sqrt * base_c * v_inv_sqrt - (lr ** 4))
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                # sample random epsilon
                noise_sample = torch.normal(mean=torch.zeros_like(d), std=noise_std)

                # update momentum (Eq 10 right in [1])
                v_momentum.add_(- (lr ** 2) * v_inv_sqrt * d - base_c * v_momentum + noise_sample)

                # update theta (Eq 10 left in [1])
                p.add_(v_momentum)
        flat_grad = torch.cat(flat_grad, dim=0)
        return loss, flat_grad


class SGHMC(SG_MCMC):
    def __init__(self, 
                params, 
                lr: float = 1e-2, 
                base_c: float = 0.05, 
                mass: float = 1, 
                #gauss_sig: float = 0.1, 
                #alpha0: float = 10, 
                #beta0: float = 10
                ):
        """
        Set up the optimizer

        :param params: Iterable, parameters serving as optimization variables
        :param lr: float, learning
        :param base_c: float, friction term
        :param mass: float, mass term
        :param gauss_sig: float, initial prior sigma
        :param alpha0: float, initial hyperprior
        :param beta0: flaot, initial hyperprior
        """
        self.eps = 1e-6
        #self.alpha0 = alpha0
        #self.beta0 = beta0
        #if gauss_sig == 0:
        #    self.weight_decay = 0
        #else:
        #    self.weight_decay = 1 / (gauss_sig ** 2)
        #if self.weight_decay <= 0.0:
        #    raise ValueError("Invalid weight_decay value: {}".format(self.weight_decay))

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_c < 0:
            raise ValueError("Invalid friction term: {}".format(base_c))
        if mass <= 0.0:
            raise ValueError("Invalid mass term: {}".format(mass))

        self.defaults = dict(
            lr=lr,
            base_c=base_c,
            mass=mass
        )
        super().__init__(params, self.defaults)

    def step(self, burn_in=None, resample_momentum=False):
        """Simulate discretized Hamiltonian dynamics for one step"""
        loss = None

        for group in self.param_groups:  # iterate over blocks -> the ones defined in defaults. We dont use groups.
            for p in group["params"]:  # these are weight and bias matrices
                if p.grad is None:
                    continue
                state = self.state[p]  # define dict for each individual param
                if len(state) == 0:
                    state["iteration"] = 0
                    state["r_momentum"] = torch.zeros_like(p)
                    #state['weight_decay'] = self.weight_decay

                state["iteration"] += 1  # this is kind of useless now but lets keep it provisionally

                #if resample_prior:
                #    state['weight_decay'] = self.resample_prior(p)

                base_c, mass, lr = group["base_c"], group["mass"], group["lr"]
                #weight_decay = state["weight_decay"]

                d_p = p.grad
                #if weight_decay != 0:
                #    d_p.add_(p.data, alpha=weight_decay)

                if resample_momentum:
                    state["r_momentum"] = torch.normal(mean=torch.zeros_like(d_p),
                                                       std=torch.sqrt(mass))
                r_momentum = state["r_momentum"]
                b = 0  # "simplest choice"
                noise_var = 2*lr(base_c - b)
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                # sample random noise
                noise_sample = torch.normal(mean=torch.zeros_like(d_p), std=noise_std)

                r_momentum.add_(- lr * d_p - lr*base_c*r_momentum/mass + noise_sample)

                p.add_(lr*r_momentum/mass)

        return loss
