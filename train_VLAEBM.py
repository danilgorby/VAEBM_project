"""Code for training VLAEBM"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision
import wandb
from PIL import Image
from torch import autograd, optim
from torch.autograd import Variable
from tqdm import tqdm

import datasets
import utils
from ebm_models import EBM_CIFAR32
from vlae_model import VAE_AF
from thirdparty.igebm_utils import clip_grad, sample_data
from utils import init_processes


def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag


def vae_sample(VAE, num_samples, eps_z=None):

    if eps_z is None:
    	eps_z = VAE.lat_dist.sample((num_samples, VAE.latent_dim)).squeeze(2)

    z = torch.zeros_like(eps_z)
    for i in range(VAE.latent_dim):
        mu_eps, log_sig_eps = VAE.made(eps_z)[:, i].chunk(2, dim=-1)
        mu_eps = mu_eps.squeeze(1)
        log_sig_eps = log_sig_eps.squeeze(1)
        z[:, i] = (eps_z[:, i] - mu_eps) / torch.exp(log_sig_eps)

    # eps = z * torch.exp(log_sig_eps) + mu_eps
    log_pz = -0.5 * torch.log(2 * torch.pi) - 0.5 * eps_z ** 2 + log_sig_eps

    logits = VAE.decoder(z)
    return logits, z, log_pz


def train(model, VAE, t, loader, opt, model_path):
    step_size = opt.step_size
    sample_step = opt.num_steps

    requires_grad(VAE.parameters(), False)
    loader = tqdm(enumerate(sample_data(loader)))

    if opt.use_wandb:
        if not os.environ.get("WANDB_API_KEY", None):
            os.environ["WANDB_API_KEY"] = opt.wandb_key

        name = opt.experiment + datetime.strftime(datetime.now(), "_%h%d_%H%M%S")
        project = "project_gans"
        wandb.init(project=project, entity="daevsikova", config=opt, name=name)

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=opt.lr, betas=(0.99, 0.999), weight_decay=opt.wd)

    if opt.use_amp:
        [model, VAE], optimizer = amp.initialize([model, VAE], optimizer, opt_level="O1")

    d_s_t = []

    # with torch.no_grad():  # get a bunch of samples to know how many groups of latent variables are there
    #     _, z_list, _ = VAE.sample(opt.batch_size, t)


    for idx, (image) in loader:
        image = image[0]
        image = image.cuda()
        bs = image.shape[0]

        noise_x = torch.randn(image.size()).cuda()
        noise_z = torch.randn((bs, 16)).cuda()

        eps_z = Variable(torch.Tensor(bs, 16).normal_(0, 1.0).cuda(), requires_grad=True)

        eps_x = torch.Tensor(image.size()).normal_(0, 1.0).cuda()
        eps_x = Variable(eps_x, requires_grad=True)

        requires_grad(parameters, False)
        model.eval()
        VAE.eval()

        # двойная динамика Ланжевена
        for k in range(sample_step):

            logits, _, log_p_total = vae_sample(VAE, opt.batch_size, eps_z)
            output = VAE.decoder_output(logits)
            neg_x = output.sample_given_eps(eps_x)

            log_pxgz = output.log_prob(neg_x).sum(dim=[1, 2, 3])

            # compute energy
            dvalue = model(neg_x) - log_p_total - log_pxgz
            dvalue = dvalue.mean()
            dvalue.backward()


            # for i in range(len(eps_z)):
            #     # update z group by group
            noise_z.normal_(0, 1)
            eps_z.data.add_(-0.5 * step_size, eps_z.grad.data * opt.batch_size)
            eps_z.data.add_(np.sqrt(step_size), noise_z.data)
            eps_z.grad.detach_()
            eps_z.grad.zero_()

            # update x
            noise_x.normal_(0, 1)
            eps_x.data.add_(-0.5 * step_size, eps_x.grad.data * opt.batch_size)
            eps_x.data.add_(np.sqrt(step_size), noise_x.data)
            eps_x.grad.detach_()
            eps_x.grad.zero_()

        eps_z = eps_z.detach()
        eps_x = eps_x.detach()

        requires_grad(parameters, True)
        model.train()

        model.zero_grad()
        logits, _, _ = vae_sample(VAE, opt.batch_size, eps_z)
        output = VAE.decoder_output(logits)

        neg_x = output.sample_given_eps(eps_x)

        pos_out = model(image)
        neg_out = model(neg_x)

        norm_loss = model.spectral_norm_parallel()  # ?
        loss_reg_s = opt.alpha_s * norm_loss

        loss = pos_out.mean() - neg_out.mean()
        loss_total = loss + loss_reg_s

        if opt.use_amp:
            with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_total.backward()

        if opt.grad_clip:
            clip_grad(model.parameters(), optimizer)

        optimizer.step()

        loader.set_description(f"loss: {loss.mean().item():.5f}")
        loss_print = pos_out.mean() - neg_out.mean()
        d_s_t.append(loss_print.item())

        wandb.log({"EMB loss": loss.mean().item(), "EMB total loss": loss_total.mean().item()})

        if idx % opt.sample_freq == 0:
            # neg_img = 0.5*output.dist.mu + 0.5
            # neg_img = 0.5*torch.sum(output.means, dim=2) + 0.5
            # neg_img = output.sample()  # _given_eps(eps_x)

            torchvision.utils.save_image(neg_x, model_path + "/images/sample.png", nrow=16, normalize=True)

            wandb.log({"Sample": wandb.Image((model_path + "/images/sample.png".format(idx)))})

            torch.save(d_s_t, model_path + "d_s_t")

        if idx % opt.save_freq == 0:
            state_dict = {}
            state_dict["model"] = model.state_dict()
            state_dict["optimizer"] = optimizer.state_dict()
            model_save_path = model_path + "EBM_{}.pth".format(idx)
            torch.save(state_dict, model_save_path)
            wandb.save(model_save_path)

        if idx == opt.total_iter:
            break


def main(eval_args):
    # ensures that weight initializations are all the same
    eval_args.save = '/content/VAEBM_project/checkpoints' # for colab
    # eval_args.save = 'checkpoints'  # for data sphere
    logging = utils.Logger(eval_args.local_rank, eval_args.save)

    # load a checkpoint
    logging.info("loading the model at:")
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location="cpu")
    args = checkpoint["args"]

    logging.info("loaded model at epoch %d", checkpoint["epoch"])
    if not hasattr(args, "ada_groups"):
        logging.info("old model, no ada groups was found.")
        args.ada_groups = False

    if not hasattr(args, "min_groups_per_scale"):
        logging.info("old model, no min_groups_per_scale was found.")
        args.min_groups_per_scale = 1

    arch_instance = utils.get_arch_cells(args.arch_instance)

    # define and load pre-trained VAE
    model = VAE_AF()
    model = model.cuda()
    print("num conv layers:", len(model.all_conv_layers))

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.cuda()

    logging.info("args = %s", args)
    logging.info("param size = %fM ", utils.count_parameters_in_M(model))

    t = 1  # temperature of VAE samples
    loader, _, num_classes = datasets.get_loaders(eval_args)

    if eval_args.dataset == "cifar10":
        EBM_model = EBM_CIFAR32(3, eval_args.n_channel, data_init=eval_args.data_init).cuda()

    model_path = "./saved_models/{}/{}/".format(eval_args.dataset, eval_args.experiment)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(model_path + "/images/")

    # use 5 batch of training images to initialize the data dependent init for weight norm
    init_image = []
    for idx, (image) in enumerate(loader):
        img = image[0]
        init_image.append(img)
        if idx == 4:
            break
    init_image = torch.stack(init_image).cuda()
    init_image = init_image.view(-1, 3, eval_args.im_size, eval_args.im_size)

    EBM_model(init_image)  # for initialization

    # call the training function
    train(EBM_model, model, t, loader, eval_args, model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("training of VAEBM")
    # experimental results
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/VAE_checkpoint.pt", help="location of the NVAE checkpoint"
    )
    parser.add_argument(
        "--experiment", default="EBM_1", help="experiment name, model checkpoint and samples will be saved here"
    )

    parser.add_argument("--save", type=str, default="/tmp/nasvae/expr", help="location of the NVAE logging")

    parser.add_argument("--dataset", type=str, default="celeba_64", help="which dataset to use")
    parser.add_argument("--im_size", type=int, default=64, help="size of image")

    parser.add_argument("--data", type=str, default="../data/celeba_64/", help="location of the data file")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate for EBM")

    # DDP.
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process")
    parser.add_argument("--world_size", type=int, default=1, help="number of gpus")
    parser.add_argument("--seed", type=int, default=1, help="seed used for initialization")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training EBM")
    parser.add_argument("--n_channel", type=int, default=64, help="initial number of channels of EBM")

    # traning parameters
    parser.add_argument("--alpha_s", type=float, default=0.2, help="spectral reg coef")

    parser.add_argument("--step_size", type=float, default=5e-6, help="step size for LD")
    parser.add_argument("--num_steps", type=int, default=10, help="number of LD steps")
    parser.add_argument("--total_iter", type=int, default=30000, help="number of training iteration")

    parser.add_argument("--wd", type=float, default=3e-5, help="weight decay for adam")
    parser.add_argument(
        "--data_init", dest="data_init", action="store_false", help="data depedent init for weight norm"
    )
    parser.add_argument(
        "--use_mu_cd",
        dest="use_mu_cd",
        action="store_true",
        help="use mean or sample from the decoder to compute CD loss",
    )
    parser.add_argument("--grad_clip", dest="grad_clip", action="store_false", help="clip the gradient as in Du et al.")
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", help="use mix precision")

    # buffer
    parser.add_argument(
        "--use_buffer", dest="use_buffer", action="store_true", help="use persistent training, default is false"
    )
    parser.add_argument("--buffer_size", type=int, default=10000, help="size of buffer")
    parser.add_argument("--max_p", type=float, default=0.6, help="maximum p of sampling from buffer")
    parser.add_argument("--anneal_step", type=float, default=5000.0, help="p annealing step")

    parser.add_argument("--comment", default="", type=str, help="some comments")

    # custom args
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--sample_freq", default=20, type=int)
    parser.add_argument("--wandb_key", default="e891f26c3ad7fd5a7e215dc4e344acc89c8861da", type=str)

    args = parser.parse_args()

    args.distributed = False
    init_processes(0, 1, main, args)