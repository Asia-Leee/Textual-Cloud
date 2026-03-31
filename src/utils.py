from copy import deepcopy
import pathlib
import yaml
import logging
import pickle
import json
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
import subprocess
import os
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from collections import defaultdict
matplotlib.use('Agg')



def read_features(path):

    with open(path, 'rb') as f:
        x = pickle.load(f)

    data = x['features']
    fps=x['fps']
    url = [str(u) for u in list(x['video_names'])]

    return data, url, fps


def fix_seeds(seed=42):

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_model_size(model, logger):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    logger.info(
        "Created network [%s] with total number of parameters: %.1f million."
        % (type(model).__name__, num_params / 1000000)
    )


def get_git_revision_hash():
    try:
        hash_string = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        hash_string = ""
    return hash_string

def dump_config_yaml(args, exp_dir):
    args_dict = deepcopy(vars(args))
    for k,v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()

    with open((exp_dir/"args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f)

def log_hparams(writer, args, metrics):
    args_dict = vars(args)
    for k,v in args_dict.items():
        if isinstance(v, pathlib.PosixPath):
            args_dict[k] = v.as_posix()
    del metrics["recall"] 
    metrics = {"Eval/"+k: v for k,v in metrics.items()}
    # del args_dict['audio_hip_blocks']
    # del args_dict['video_hip_blocks']

    writer.add_hparams(args_dict, metrics)


def setup_experiment(args, stage, *stats):

    state_name= f"(epoch{args.epochs_stage1}_bs{args.bs}_lr{args.lr}_nbatches{args.n_batches})"
    exp_name=f"({args.exp_name})_{datetime.now().strftime('%b-%d_%H-%M')}_{stage}"

    exp_dir = (args.log_dir/Path(state_name)/Path(exp_name))

    exp_dir.mkdir(parents=True)
    pickle.dump(args, (exp_dir / "args.pkl").open("wb")) 

    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = create_logger(exp_dir / "train.log")

    logger.info("\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())))
    logger.info(f"The experiment will be stored in {exp_dir.resolve()}\n")

    return logger, exp_dir,



def setup_visualizations(args, *stats):

    eval_dir = args.load_path_stage_B
    assert eval_dir.exists()

    # test_stats = PD_Stats(eval_dir / "test_stats.pkl", list(sorted(stats)))
    test_stats = PD_Stats(eval_dir / "test_stats.pkl", ['seen', 'unseen', 'hm', 'zsl'])
    logger = create_logger(eval_dir / "visuals.log")

    logger.info(f"Start visualizing {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    # for Tensorboard hparam logging
    writer = SummaryWriter(log_dir=eval_dir)

    return logger, eval_dir, test_stats, writer


def load_model_parameters(model, model_weights):
    logger = logging.getLogger()
    loaded_state = model_weights
    self_state = model.state_dict()
    for name, param in loaded_state.items():
        param = param
        if 'module.' in name:
            name = name.replace('module.', '')
        if name in self_state.keys():
            self_state[name].copy_(param)
        else:
            logger.info("didnt load ", name)


def load_args(path):
    return pickle.load((path / "args.pkl").open("rb"))


def cos_dist(a, b):
    # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return res



def get_class_names(path):
    if isinstance(path, str):
        path = Path(path)
    with path.open("r") as f:
        classes = sorted([line.strip() for line in f])
    return classes


def load_model_weights(weights_path, model):
    logging.info(f"Loading model weights from {weights_path}")
    load_dict = torch.load(weights_path)
    model_weights = load_dict["model"]
    epoch = load_dict["epoch"]
    logging.info(f"Load from epoch: {epoch}")
    load_model_parameters(model, model_weights)
    return epoch

def plot_hist_from_dict(dict):
    plt.bar(range(len(dict)), list(dict.values()), align="center")
    plt.xticks(range(len(dict)), list(dict.keys()), rotation='vertical')
    plt.tight_layout()
    plt.show()

def save_class_performances(seen_dict, unseen_dict, dataset_name, args=None):
    roor_path = args.load_path_stage_B
    seen_dir = os.path.join(roor_path, f'class_performance_{dataset_name}_seen.pkl')
    unseen_dir = os.path.join(roor_path, f'class_performance_{dataset_name}_unseen.pkl')

    seen_path = Path(seen_dir)
    unseen_path = Path(unseen_dir)
    with seen_path.open("wb") as f:
        pickle.dump(seen_dict, f)
        logging.info(f"Saving seen class performances to {seen_path}")
    with unseen_path.open("wb") as f:
        pickle.dump(unseen_dict, f)
        logging.info(f"Saving unseen class performances to {unseen_path}")


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)






class SamplerFactory:
    """
    Factory class to create balanced samplers.
    https://github.com/khornlund/pytorch-balanced-sampler
    """

    def __init__(self, logger, verbose=0):
        self.logger = logger

    def get(self, class_idxs, batch_size, n_batches, alpha, kind):
        """
        Parameters
        ----------
        class_idxs : 2D list of ints
            List of sample indices for each class. Eg. [[0, 1], [2, 3]] implies indices 0, 1
            belong to class 0, and indices 2, 3 belong to class 1.

        batch_size : int
            The batch size to use.

        n_batches : int
            The number of batches per epoch.

        alpha : numeric in range [0, 1]
            Weighting term used to determine weights of each class in each batch.
            When `alpha` == 0, the batch class distribution will approximate the training population
            class distribution.
            When `alpha` == 1, the batch class distribution will approximate a uniform distribution,
            with equal number of samples from each class.

        kind : str ['fixed' | 'random']
            The kind of sampler. `Fixed` will ensure each batch contains a constant proportion of
            samples from each class. `Random` will simply sample with replacement according to the
            calculated weights.
        """
        if kind == 'random':
            return self.random(class_idxs, batch_size, n_batches, alpha)
        if kind == 'fixed':
            return self.fixed(class_idxs, batch_size, n_batches, alpha)
        raise Exception(f'Received kind {kind}, must be `random` or `fixed`')

    def random(self, class_idxs, batch_size, n_batches, alpha):
        self.logger.info(f'Creating WeightedRandomBatchSampler...')
        class_sizes, weights = self._weight_classes(class_idxs, alpha) 
        sample_rates = self._sample_rates(weights, class_sizes) 
        return WeightedRandomBatchSampler(sample_rates, class_idxs, batch_size, n_batches)

    def fixed(self, class_idxs, batch_size, n_batches, alpha):
        self.logger.info(f'Creating WeightedFixedBatchSampler...')
        class_sizes, weights = self._weight_classes(class_idxs, alpha)
        class_samples_per_batch = self._fix_batches(weights, class_sizes, batch_size, n_batches)
        return WeightedFixedBatchSampler(class_samples_per_batch, class_idxs, n_batches)

    def _weight_classes(self, class_idxs, alpha):
        class_sizes = np.asarray([len(idxs) for idxs in class_idxs]) 
        n_samples = class_sizes.sum()
        n_classes = len(class_idxs)

        original_weights = np.asarray([size / n_samples for size in class_sizes]) 
        uniform_weights = np.repeat(1 / n_classes, n_classes) 

        self.logger.info(f'Sample population absolute class sizes: {class_sizes}')
        self.logger.info(f'Sample population relative class sizes: {original_weights}')

        weights = self._balance_weights(uniform_weights, original_weights, alpha)
        return class_sizes, weights

    def _balance_weights(self, weight_a, weight_b, alpha):
        assert alpha >= 0 and alpha <= 1, f'invalid alpha {alpha}, must be 0 <= alpha <= 1'
        beta = 1 - alpha
        weights = (alpha * weight_a) + (beta * weight_b)
        self.logger.info(f'Target batch class distribution {weights} using alpha={alpha}')
        return weights

    def _sample_rates(self, weights, class_sizes):
        return weights / class_sizes 

    def _fix_batches(self, weights, class_sizes, batch_size, n_batches):
        """
        Calculates the number of samples of each class to include in each batch, and the number
        of batches required to use all the data in an epoch.
        """
        class_samples_per_batch = np.round((weights * batch_size)).astype(int)

        # cleanup rounding edge-cases
        remainder = batch_size - class_samples_per_batch.sum()
        largest_class = np.argmax(class_samples_per_batch)
        class_samples_per_batch[largest_class] += remainder

        assert class_samples_per_batch.sum() == batch_size

        proportions_of_class_per_batch = class_samples_per_batch / batch_size
        self.logger.info(f'Rounded batch class distribution {proportions_of_class_per_batch}')

        proportions_of_samples_per_batch = class_samples_per_batch / class_sizes

        self.logger.info(f'Expecting {class_samples_per_batch} samples of each class per batch, '
                         f'over {n_batches} batches of size {batch_size}')

        oversample_rates = proportions_of_samples_per_batch * n_batches
        self.logger.info(f'Sampling rates: {oversample_rates}')

        return class_samples_per_batch


class WeightedRandomBatchSampler(BatchSampler):
    """
    Samples with replacement according to the provided weights.

    Parameters
    ----------
    class_weights : `numpy.array(int)`
        The number of samples of each class to include in each batch.

    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.

    batch_size : int
        The size of each batch yielded.

    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_weights, class_idxs, batch_size, n_batches):
        self.sample_idxs = []
        for idxs in class_idxs:  
            self.sample_idxs.extend(idxs)

        sample_weights = [] 
        for c, weight in enumerate(class_weights):
            sample_weights.extend([weight] * len(class_idxs[c]))

        self.sampler = WeightedRandomSampler(sample_weights, batch_size, replacement=True) 
        self.n_batches = n_batches

    def __iter__(self):
        for bidx in range(self.n_batches): 
            selected = []
            for idx in self.sampler: 
                selected.append(self.sample_idxs[idx])
            yield selected

    def __len__(self):
        return self.n_batches


class WeightedFixedBatchSampler(BatchSampler):
    """
    Ensures each batch contains a given class distribution.

    The lists of indices for each class are shuffled at the start of each call to `__iter__`.

    Parameters
    ----------
    class_samples_per_batch : `numpy.array(int)`
        The number of samples of each class to include in each batch.

    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.

    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_samples_per_batch, class_idxs, n_batches):
        self.class_samples_per_batch = class_samples_per_batch
        self.class_idxs = [CircularList(idx) for idx in class_idxs]
        self.n_batches = n_batches

        self.n_classes = len(self.class_samples_per_batch)
        self.batch_size = self.class_samples_per_batch.sum()

        assert len(self.class_samples_per_batch) == len(self.class_idxs)
        assert isinstance(self.n_batches, int)

    def _get_batch(self, start_idxs):
        selected = []
        for c, size in enumerate(self.class_samples_per_batch):
            selected.extend(self.class_idxs[c][start_idxs[c]:start_idxs[c] + size])
        np.random.shuffle(selected)
        return selected

    def __iter__(self):
        [cidx.shuffle() for cidx in self.class_idxs]
        start_idxs = np.zeros(self.n_classes, dtype=int)
        for bidx in range(self.n_batches):
            yield self._get_batch(start_idxs)
            start_idxs += self.class_samples_per_batch

    def __len__(self):
        return self.n_batches


class CircularList:
    """
    Applies modulo function to indexing.
    """

    def __init__(self, items):
        self._items = items
        self._mod = len(self._items)
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self._items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(key.start, key.stop)]
        return self._items[key % self._mod]

def save_json_file(data, path):
    if isinstance(data, defaultdict):
        with open(path, 'w') as f:
            json.dump(dict(data), f)
    else:
        with open(path, 'w') as f:
            json.dump(data, f)


def visuallize_loss_curve_from_json(
        json_path_train_loss,
        json_path_val_loss,
        hm_scores=None,
        visual_lr_epochs=None,
        save_path=None):
    
    tick_label_size = 15  
    legend_font_size = 23  
    title_font_size = 14  

    with open(json_path_train_loss, 'r') as f:
        train_loss = json.load(f)
    with open(json_path_val_loss, 'r') as f:
        val_loss = json.load(f)

    keys = list(train_loss.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, key in zip(axes, keys):
        train_loss_list = [x for ep in train_loss[key] for x in ep]
        val_loss_list = [x for ep in val_loss[key] for x in ep] if key in val_loss else []

        ax.plot(train_loss_list, label='Train ', linewidth=1, alpha=0.8)
        ax.plot(val_loss_list, label='Val ', linewidth=1, alpha=0.8)

        if len(train_loss[key]) > 0:
            steps_per_ep = len(train_loss[key][0])
            n_eps = len(train_loss[key])

            ticks = [(i + 1) * steps_per_ep for i in range(n_eps)]

            ax.set_xticks(ticks)
            
            ax.set_xticklabels([f'{i + 1}' for i in range(n_eps)], fontsize=tick_label_size)

            for t in ticks: ax.axvline(x=t, color='k', linestyle='--', alpha=0.1)
            if key == 'total_loss' and hm_scores:
                ax2 = ax.twinx()

                ax2.plot(ticks, hm_scores, color='black', linestyle='-', linewidth=1, label='HM Score')
                ax2.set_ylabel('HM Score', color='black', fontsize=tick_label_size)  
                
                ax2.tick_params(axis='y', labelsize=tick_label_size)

        
        ax.set_title(key, fontweight='bold', fontsize=title_font_size)

        
        if key == 'total_loss' and hm_scores:
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=legend_font_size)
        else:
            ax.legend(fontsize=legend_font_size)

        ax.grid(True, alpha=0.2)
        
        ax.tick_params(axis='y', labelsize=tick_label_size)

    if visual_lr_epochs is not None and len(keys) < len(axes):
        lr_ax = axes[len(keys)]
        lr_flat = [x for ep in visual_lr_epochs for x in ep]
        lr_ax.plot(lr_flat, label='Learning Rate', color='black', linewidth=1, alpha=0.8)
        steps_per_ep_lr = len(visual_lr_epochs[0])

        n_eps_lr = len(visual_lr_epochs)
        ticks_lr = [(i + 1) * steps_per_ep_lr for i in range(n_eps_lr)]
        lr_ax.set_xticks(ticks_lr)
        
        lr_ax.set_xticklabels([f'{i + 1}' for i in range(n_eps_lr)], fontsize=tick_label_size)

        for t in ticks_lr: lr_ax.axvline(x=t, color='k', linestyle='--', alpha=0.1)

        lr_ax.set_title("Learning Rate", fontweight='bold', fontsize=title_font_size)
        
        lr_ax.legend(fontsize=legend_font_size)
        lr_ax.grid(True, alpha=0.2)
        lr_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        
        lr_ax.tick_params(axis='y', labelsize=tick_label_size)
        
        lr_ax.yaxis.get_offset_text().set_fontsize(tick_label_size)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
