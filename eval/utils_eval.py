from src.utils import save_class_performances,PD_Stats,create_logger,load_args
from torch.utils.tensorboard import SummaryWriter


def setup_evaluation(args,):

    eval_dir = args.load_path_stage_B 
    assert eval_dir.exists()

    logger = create_logger(eval_dir / "eval.log")

    logger.info(f"Start evaluation {eval_dir}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"Loaded configuration {args.load_path_stage_B / 'args.pkl'}")
    logger.info(
        "\n".join(f"{k}: {str(v)}" for k, v in sorted(dict(vars(load_args(args.load_path_stage_B))).items()))
    )
    logger.info(f"The evaluation will be stored in {eval_dir.resolve()}\n")
    logger.info("")

    return logger, eval_dir,