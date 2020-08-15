import torch
import copy
import numpy as np
import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class BNNTrainer(object):
    def __init__(self, model, optimizer, nll_func, err_func, trainloader, valloader, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.nll_func = nll_func
        self.err_func = err_func
        self.trainloader = trainloader
        self.valloader = valloader

        self.resample_prior_every = kwargs.get('resample_prior_every', 100)
        self.resample_momentum_every = kwargs.get('resample_momentum_every', 50)
        device = kwargs.get('device', 'cuda:0')
        self.max_weight_set_size = kwargs.get('max_weight_set_size', 200)
        self.report_every = kwargs.get('report_every', 100)
        self.save_freq = kwargs.get('save_freq', 2)
        self.batch_size = kwargs.get('batch_size', 500)
        #self.resample_prior_until = kwargs.get('resample_prior_intil', 100)
        #self.burn_in_steps = kwargs.get('burn_in_steps', 200)
        
        self.device = torch.device(device)
        self.N_train = len(trainloader) * self.batch_size
        self.weight_set_samples = []
        self.early_stopping = kwargs.get('early_stopping', False)
        if self.early_stopping:
            self.min_delta = kwargs.get('min_delta', 1e-3)
            self.wait = kwargs.get('wait', 10)

        
    def train(self, n_epoch, burn_in_epochs=0, resample_prior_until=1e8):
        best_loss = 1e9
        no_significant_improvement_step = 0
        self.model.to(self.device)
        it_cnt = 0
        for epoch in range(n_epoch):
            train_loss, val_loss = 0, 0
            train_err, val_err = 0, 0
            n_ex = 0
            self.model.train()
            burn_in = epoch < burn_in_epochs
            for x, y in self.trainloader:
                resample_prior = it_cnt % self.resample_prior_every == 0 and epoch < resample_prior_until
                resample_momentum = it_cnt % self.resample_momentum_every == 0
                loss, err = self.do_train_step(x.to(self.device), y.to(self.device), 
                    resample_prior=resample_prior, resample_momentum=resample_momentum, burn_in=burn_in)
                #if resample_prior:
                #    self.weight_set_samples = []
                train_loss += loss
                train_err += err
                n_ex += x.shape[0]
                it_cnt += 1

            if epoch > burn_in_epochs and epoch % self.save_freq == 0:
                self.save_sampled_net()

                #resample_prior = False
                #resample_momentum = False

                #loc_iter = n_iter % self.N_train
                #if (loc_iter + 1) % self.report_every == 0:
                #    logger.info(f'Iteration {n_iter}, Loss {train_loss /  n_ex}, Error {train_err / n_ex}')
            
            n_ex = 0
            self.model.eval()
            for x, y in self.valloader:
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    val_err += self.err_func(y_hat, y).sum().item()
                    val_loss += self.nll_func(y_hat, y)
                    n_ex += x.shape[0]
            logger.info(f'Epoch {epoch} finished. Val loss {val_loss / n_ex}, Val error {val_err / n_ex}')
            if self.early_stopping:
                if val_loss / n_ex - best_loss < -self.min_delta:
                    best_loss = val_loss / n_ex
                else:
                    no_significant_improvement_step += 1

                if no_significant_improvement_step >= self.wait:
                    logger.info('Early Stopping triggered')
                    break

    def do_train_step(self, x, y, **kwargs):
        y_hat = self.model(x)
        nll = self.nll_func(y_hat, y)
        nll *= self.N_train / x.shape[0]
        self.optimizer.zero_grad() 
        nll.backward()
        self.optimizer.step()#**kwargs)

        err = self.err_func(y_hat, y).sum().item()

        return nll, err

    def get_weight_samples(self, Nsamples=0):
        """return weight samples from posterior in a single-column array"""
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        # for idx, state_dict in enumerate(self.weight_set_samples):
        #     if idx == Nsamples:
        #         break

        #     for key in state_dict.keys():
                # if 'weight' in key:
                #     weight_mtx = state_dict[key].cpu().data
                #     for weight in weight_mtx.view(-1):
                #         weight_vec.append(weight)

        return self.weight_set_samples

    def save_sampled_net(self):

        max_samples = self.max_weight_set_size
        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        #cprint('c', ' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples))
        return None

    # def save_weights(self, filename):
    #     save_object(self.weight_set_samples, filename)

    # def load_weights(self, filename, subsample=1):
    #     self.weight_set_samples = load_object(filename)
    #     self.weight_set_samples = self.weight_set_samples[::subsample]

# class BNN_cat(BaseNet):  # for categorical distributions
#     def __init__(self, N_train, lr=1e-2, cuda=True, grad_std_mul=30):
#         super(BNN_cat, self).__init__()

#         cprint('y', 'BNN categorical output')
#         self.lr = lr
#         self.model = MLP(input_dim=784, width=1200, depth=2, output_dim=10)
#         self.cuda = cuda

#         self.N_train = N_train
#         self.create_net()
#         self.create_opt()
#         self.schedule = None  # [] #[50,200,400,600]
#         self.epoch = 0

#         self.grad_buff = []
#         self.max_grad = 1e20
#         self.grad_std_mul = grad_std_mul

#         self.weight_set_samples = []

#     def create_net(self):
#         torch.manual_seed(42)
#         if self.cuda:
#             torch.cuda.manual_seed(42)
#         if self.cuda:
#             self.model.cuda()

#         print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

#     def create_opt(self):
#         """This optimiser incorporates the gaussian prior term automatically. The prior variance is gibbs sampled from
#         its posterior using a gamma hyper-prior."""
#         self.optimizer = H_SA_SGHMC(params=self.model.parameters(), lr=self.lr, base_C=0.05, gauss_sig=0.1)  # this last parameter does nothing

#     def fit(self, x, y, burn_in=False, resample_momentum=False, resample_prior=False):
#         self.set_mode_train(train=True)
#         x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
#         self.optimizer.zero_grad()
#         out = self.model(x)
#         loss = F.cross_entropy(out, y, reduction='mean')
#         loss = loss * self.N_train  # We use mean because we treat as an estimation of whole dataset
#         loss.backward()

#         # Gradient buffer to allow for dynamic clipping and prevent explosions
#         if len(self.grad_buff) > 1000:
#             self.max_grad = np.mean(self.grad_buff) + self.grad_std_mul * np.std(self.grad_buff)
#             self.grad_buff.pop(0)
#         # Clipping to prevent explosions
#         self.grad_buff.append(nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
#                                                        max_norm=self.max_grad, norm_type=2))
#         if self.grad_buff[-1] >= self.max_grad:
#             print(self.max_grad, self.grad_buff[-1])
#             self.grad_buff.pop()
#         self.optimizer.step(burn_in=burn_in, resample_momentum=resample_momentum, resample_prior=resample_prior)

#         # out: (batch_size, out_channels, out_caps_dims)
#         pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
#         err = pred.ne(y.data).sum()

#         return loss.data * x.shape[0] / self.N_train, err

#     def eval(self, x, y, train=False):
#         self.set_mode_train(train=False)
#         x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

#         out = self.model(x)
#         loss = F.cross_entropy(out, y, reduction='sum')
#         probs = F.softmax(out, dim=1).data.cpu()

#         pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
#         err = pred.ne(y.data).sum()

#         return loss.data, err, probs

#     def save_sampled_net(self, max_samples):

#         if len(self.weight_set_samples) >= max_samples:
#             self.weight_set_samples.pop(0)

#         self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

#         cprint('c', ' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples))
#         return None

#     def predict(self, x):
#         self.set_mode_train(train=False)
#         x, = to_variable(var=(x, ), cuda=self.cuda)
#         out = self.model(x)
#         probs = F.softmax(out, dim=1).data.cpu()
#         return probs.data

#     def sample_predict(self, x, Nsamples=0, grad=False):
#         """return predictions using multiple samples from posterior"""
#         self.set_mode_train(train=False)
#         if Nsamples == 0:
#             Nsamples = len(self.weight_set_samples)
#         x, = to_variable(var=(x, ), cuda=self.cuda)

#         if grad:
#             self.optimizer.zero_grad()
#             if not x.requires_grad:
#                 x.requires_grad = True

#         out = x.data.new(Nsamples, x.shape[0], self.model.output_dim)

#         # iterate over all saved weight configuration samples
#         for idx, weight_dict in enumerate(self.weight_set_samples):
#             if idx == Nsamples:
#                 break
#             self.model.load_state_dict(weight_dict)
#             out[idx] = self.model(x)

#         out = out[:idx]
#         prob_out = F.softmax(out, dim=2)

#         if grad:
#             return prob_out
#         else:
#             return prob_out.data

#     def get_weight_samples(self, Nsamples=0):
#         """return weight samples from posterior in a single-column array"""
#         weight_vec = []

#         if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
#             Nsamples = len(self.weight_set_samples)

#         for idx, state_dict in enumerate(self.weight_set_samples):
#             if idx == Nsamples:
#                 break

#             for key in state_dict.keys():
#                 if 'weight' in key:
#                     weight_mtx = state_dict[key].cpu().data
#                     for weight in weight_mtx.view(-1):
#                         weight_vec.append(weight)

#         return np.array(weight_vec)

#     def save_weights(self, filename):
#         save_object(self.weight_set_samples, filename)

#     def load_weights(self, filename, subsample=1):
#         self.weight_set_samples = load_object(filename)
#         self.weight_set_samples = self.weight_set_samples[::subsample]



