# [VAEBM]((https://arxiv.org/abs/2010.00654)): A Symbiosis between VAE and EBM

VAEBM обучает энергию (energy network) для уточнения распределения данных, полученных NVAE, 
где энергия и VAE совместно определяют EBM. 

В обучении VAEBM выделяется два этапа:
* обучение NVAE
* обучения энергии

## Set up datasets ##
We trained on several datasets, including CIFAR10, CelebA64, LSUN Church 64 and CelebA HQ 256. 
For large datasets, we store the data in LMDB datasets for I/O efficiency. Check [here](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) for information regarding dataset preparation.


## Training NVAE ##
#### CIFAR-10 (8x 16-GB GPUs) ####
```
python train.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
      --num_channels_enc 128 --num_channels_dec 128 --epochs 400 --num_postprocess_cells 2 --num_preprocess_cells 2 \
      --num_latent_scales 1 --num_latent_per_group 20 --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
      --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale 30 --batch_size 32 \
      --weight_decay_norm 1e-1 --num_nf 1 --num_mixture_dec 1 --fast_adamax  --arch_instance res_mbconv \
      --num_process_per_node 8 --use_se --res_dist
```


## Training VAEBM ##
We use the following commands on each dataset for training VAEBM. 
Note that you need to train the NVAE on corresponding dataset before 
running the training command here.
After training the NVAE, pass the path of the checkpoint to the `--checkpoint` argument.

#### CIFAR-10 (1x 32-GB GPU)

```
python train_VAEBM.py  --checkpoint ./checkpoints/cifar10/checkpoint.pt --experiment cifar10_exp1
--dataset cifar10 --im_size 32 --data ./data/cifar10 --num_steps 10 
--wd 3e-5 --step_size 8e-5 --total_iter 30000 --alpha_s 0.2 --lr 4e-5 --max_p 0.6 
--anneal_step 5000. --batch_size 32 --n_channel 128
```


## Sampling from VAEBM ##
To generate samples from VAEBM after training, run ```sample_VAEBM.py```, 
and it will generate 50000 test images in your given path. 
When sampling, we typically use 
longer Langvin dynamics than training for better sample quality.

```
python sample_VAEBM.py --checkpoint ./checkpoints/cifar_10/checkpoint.pt --ebm_checkpoint ./saved_models/cifar_10/cifar_exp1/EBM.pth 
--dataset cifar10 --im_size 32 --batch_size 40 --n_channel 128 --num_steps 16 --step_size 8e-5 
```


## Evaluation ##
After sampling, use the [Tensorflow](https://github.com/bioinf-jku/TTUR) or [PyTorch](https://github.com/mseitzer/pytorch-fid) 
implementation to compute the FID scores. For example, when using the Tensorflow implementation, you can obtain the FID score by saving the training images in ```/path/to/training_images``` and running the script:
```
python fid.py /path/to/training_images /path/to/sampled_images
```

For CIFAR-10, the training statistics can be downloaded from [here](https://github.com/bioinf-jku/TTUR#precalculated-statistics-for-fid-calculation), and the FID score can be computed by running
```
python fid.py /path/to/sampled_images /path/to/precalculated_stats.npz
```

For the Inception Score, save samples in a single numpy array with pixel values in range [0, 255] and simply run 
```
python ./thirdparty/inception_score.py --sample_dir /path/to/sampled_images
```
where the code for computing Inception Score is adapted from [here](https://github.com/tsc2017/Inception-Score).
