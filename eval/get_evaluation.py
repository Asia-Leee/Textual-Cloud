import copy
from dataset_all.dataset_util import DefaultCollator
from src.args import args_main
from torch.utils import data
from dataset_all.VGGSound_ZSL import VGGSoundDataset
from dataset_all.UCF_GZSL import UCFDataset
from dataset_all.ActivityNet_GZSL import ActivityNetDataset
from dataset_all.dataset_util import ContrastiveDataset
from model.my_zero_model import My_zero_shot_model
from eval.test import test
from src.utils import fix_seeds, load_args , load_model_weights, log_hparams, print_model_size
from .utils_eval import setup_evaluation


def get_evaluation(args):

    config = load_args(args.load_path_stage_B)  
    config.root_idr = args.root_dir 

    fix_seeds(config.seed)

    
    logger, eval_dir, = setup_evaluation(args, )

    if args.dataset_name == "AudioSetZSL":
        val_all_dataset = AudioSetZSLDataset(args=config,dataset_split="val",zero_shot_mode="all",)
        test_dataset = AudioSetZSLDataset(args=config,dataset_split="test",zero_shot_mode="all",)
    elif args.dataset_name == "VGGSound":
        val_dataset = VGGSoundDataset(args=config,dataset_split="val")
        test_dataset = VGGSoundDataset(args=config,dataset_split="test")
    elif args.dataset_name == "UCF":
        val_dataset = UCFDataset(args=config,dataset_split="val",)
        test_dataset = UCFDataset(args=config,dataset_split="test",)
    elif args.dataset_name == "ActivityNet":
        val_dataset = ActivityNetDataset(args=config,dataset_split="val")
        test_dataset = ActivityNetDataset(args=config,dataset_split="test",)
    else:
        raise NotImplementedError()


    contrastive_val_dataset = ContrastiveDataset(val_dataset)
    contrastive_test_dataset = ContrastiveDataset(test_dataset)

    collator_test = DefaultCollator(mode=args.batch_seqlen_test, max_len=args.batch_seqlen_test_maxlen, trim=args.batch_seqlen_test_trim)

    val_loader = data.DataLoader(
        dataset=contrastive_val_dataset,
        collate_fn=collator_test,
        batch_size=args.eval_bs,
        num_workers=args.num_workers,
    )

    test_loader = data.DataLoader(
        dataset=contrastive_test_dataset,
        collate_fn=collator_test,
        batch_size=args.eval_bs,
        num_workers=args.num_workers,
    )

    model_A = My_zero_shot_model(config)
    model_B = copy.deepcopy(model_A)

    weights_path_stage_A = list(args.load_path_stage_A.glob(f"*_{config.best_model_criterion}.pt"))[0] 
    _ = load_model_weights(weights_path_stage_A, model_A)
    weights_path_stage_B = list((args.load_path_stage_B).glob(f"*_{config.best_model_criterion}.pt"))[0]
    _ = load_model_weights(weights_path_stage_B, model_B)

    model_A.to(config.device)
    model_B.to(config.device)

    test(
        val_dataset=val_dataset,
        val_loader=val_loader,
        test_dataset=test_dataset,
        test_loader=test_loader,
        model_A=model_A,
        model_B=model_B,
        device=args.device,
        distance_fn=config.distance_fn,
        args=config,
    )

    logger.info("FINISHED")



if __name__ == "__main__":
    args, eval_args = args_main()
    get_evaluation(eval_args)
