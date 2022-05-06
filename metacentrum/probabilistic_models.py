import torch
import torch.nn as nn
from torch.optim import Adam

from distributions.distribution_wrappers import ProbabilisticWrapper
from torch.distributions.normal import Normal
from models.simple_model import SimpleModel

from training.distribution_trainer import NLLSingleDistributionTrainer
from distributions.distribution_wrappers import GaussianEnsembleWrapper

from distributions.nw_prior import NormalWishartPrior
from training.distillation_trainer import DistillationTrainer



@torch.no_grad()
def get_information_basic_model(model, x_test, y_test):
    info = {}
    output = model(torch.tensor(x_test))
    info['mean'] = output.mean.squeeze().squeeze().cpu().numpy()
    info['entropy'] = output.entropy().squeeze().cpu().numpy()
    info['stddev'] = output.stddev.squeeze().cpu().numpy()
    return info

@torch.no_grad()
def get_information_ens(model, x_test, y_test):
    info = {}
    output = model(torch.tensor(x_test))
    info['mean'] = output.expected_mean().squeeze().cpu().numpy()
    info['data_uncertainty'] = np.sqrt( output.expected_variance().squeeze().cpu().numpy() )
    info['knowledge_uncertainty'] = np.sqrt( output.variance_of_expected().squeeze().cpu().numpy() )
    
    for i in ['expected_entropy', 'expected_pairwise_kl', 'total_variance']:
        info[i] = getattr(output, i)().squeeze().cpu().numpy()
    return info

@torch.no_grad()
def get_information_distillation(model, x_test, y_test):
    info = {}
    output = model(torch.tensor(x_test))
    info['mean'] = output.mean.squeeze().squeeze().cpu().numpy()
    info['entropy'] = output.entropy().squeeze().cpu().numpy()
    info['stddev'] = output.stddev.squeeze().cpu().numpy()
    return info

@torch.no_grad()
def get_information_double_distillation(model, x_test, y_test):
    info = {}
    output = model(torch.tensor(x_test))
    info['mean'] = output.mean.squeeze().cpu().numpy()
    info['data_uncertainty'] = np.sqrt( output.expected_variance().squeeze().cpu().numpy() )
    info['knowledge_uncertainty'] = np.sqrt( output.variance_of_expected().squeeze().cpu().numpy() )
    
    # extra
    for i in [  'predictive_posterior_entropy', 'mutual_information',
                'expected_entropy', 'expected_pairwise_kl', 'total_variance',
                'predictive_posterior_variance']:
        info[i] = getattr(output, i)().squeeze().cpu().numpy()
    return info

@torch.no_grad()
def extend_info(prefix, func, model, x_test, y_test):
    info = func(model, x_test, y_test)
    info['MSE'] = np.mean(np.square(info['mean'] - y_test))
    info['MAE'] = np.mean(np.abs(info['mean'] - y_test))
    info['RDE'] = LossRDE_auto()(info['mean'], y_test)

    return {prefix + '_' + key : value for key, value in info.items()}


#################################################################
#################################################################
#################################################################


class ToyNLLTrainer(NLLSingleDistributionTrainer):
    def logging_step(self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
        ):
        pass
    def eval_step(self, val_loader, current_step, current_epoch):
        pass
    
class ToyDistillationTrainer(DistillationTrainer):
    def logging_step(self, val_loader, current_step,
        current_epoch, step_idx, steps_per_epoch
    ):
        pass
    def eval_step(self, val_loader, current_step, current_epoch):
        pass
    

#################################################################
#################################################################
#################################################################

def get_number_of_features_from_dataset(dataset):
    Nfeatures = dataset.dataset.tensors[0].shape[1]
    #print(dataset)
    #Nfeatures = dataset.dataset[0].shape[1]
    return Nfeatures


def generate_model(input_dim, out_channels=2):
    return SimpleModel(input_dim, 1, 200, 
                num_hidden = 6, 
                out_channels = out_channels,
                activation = nn.ELU
            )
# model ...
def train_NLL_model(train_dataloader, test_dataloader):
    Nfeatures = get_number_of_features_from_dataset(train_dataloader)
    
    # model
    model = generate_model(Nfeatures)
    model = ProbabilisticWrapper(Normal, model)
    
    # train
    trainer = ToyNLLTrainer(
        model, Adam, lambda logdir: logdir, 'none',
        600, {"lr": 1e-3, "warmup_steps": 600}
    )
    
    for a in train_dataloader:
        print(a)
    
    trainer.train(train_dataloader, test_dataloader)
    return model

# ensemble ...
def train_NLL_ensemble(train_dataloader, test_dataloader, n=5):
    ensemble = []
    for i in range(n):
        model = train_NLL_model(train_dataloader, test_dataloader)
        ensemble.append(model)
    return ensemble


#################################################################
#################################################################
#################################################################


# ensemble ...
def train_ensemble(ensemble):
    ensemble_model = GaussianEnsembleWrapper(
        [x.model for x in ensemble])
    return ensemble_model

# distilation ...
def train_distillation(ensemble, train_dataloader, test_dataloader):
    Nfeatures = get_number_of_features_from_dataset(train_dataloader)

    ens_model = train_ensemble(ensemble)
    
    model = generate_model(Nfeatures, out_channels=2)
    end_model = ProbabilisticWrapper(Normal, model)
    
    trainer = ToyDistillationTrainer(
        ens_model, 10.0, end_model, Adam, lambda logdir: logdir, 
        'none', 600, {"lr": 1e-3, "warmup_steps": 600}
    )
    trainer.train(train_dataloader, test_dataloader)
    return end_model

# double distilation ...
def train_double_distillation(ensemble, train_dataloader, test_dataloader):
    Nfeatures = get_number_of_features_from_dataset(train_dataloader)

    ens_model = train_ensemble(ensemble)
    
    model = generate_model(Nfeatures, out_channels=3)
    endd_model = ProbabilisticWrapper(NormalWishartPrior, model)
    
    trainer = ToyDistillationTrainer(
        ens_model, 10.0, endd_model, Adam, lambda logdir: logdir, 
        'none', 600, {"lr": 1e-3, "warmup_steps": 600}
    )
    trainer.train(train_dataloader, test_dataloader)
    return endd_model


#################################################################
#################################################################
#################################################################




#################################################################
#################################################################
#################################################################