import configargparse 
import pathlib
import yaml


def args_main(*args, **kwargs):
    parser = configargparse.ArgParser(
        description="Explainable Audio Visual Low Shot Learning",
        default_config_files=["config/my_zero_shot.yaml"], 
        config_file_parser_class=configargparse.YAMLConfigFileParser, 
        ignore_unknown_config_file_keys=True,
    )

    # parser.add_argument('-c', '--cfg', default="config/my_zero_shot.yaml", is_config_file=True, help='config file path')

    parser.add_argument('--run', default='all', type=str, choices=['all', 'stage-1', 'stage-2', 'eval'])




    parser.add_argument( '--hidden_size', type=int, default=512, )
    parser.add_argument( '--dropout', type=float, default=0.1, )
    parser.add_argument( '--mininterval', type=float,default=0.1, )
    parser.add_argument( '--epochs_stage1', type=int,default=15, )
    parser.add_argument( '--epochs_stage2' ,type=int,default=-1, )
    parser.add_argument( '--best_model_criterion', type=str,default="score", )
    parser.add_argument( '--AV_coefficient', type=float,default=0.1 )
    parser.add_argument( '--context_coefficient', type=float,default=0.1 )
    parser.add_argument( '--weight_decay', type=float,default=1e-5 )
    parser.add_argument( '--aug_noise_level', type=float, default=0.0 )

    #textual transformer model
    parser.add_argument( '--input_dim', type=int, default=512 )
    parser.add_argument( '--nhead', type=int, default=8 )
    parser.add_argument( '--dim_feedforward', type=int, default=2048 )
    parser.add_argument( '--num_layers', type=int, default=1 )


    parser.add_argument(
        "--modality",
        help="Wether to use embeddings  audio features, video features, or both combined",
        choices=["video", "audio", "both"],
        default='both',
        type=str,
        # required=True
    )


    parser.add_argument(
        "--data_dir",
        help="Path to dataset directory. Expected subfolder structure: '{root_dir}/features/{feature_extraction_method}/{audio,video,text}'",
        default="/Dataset/VGGSound/",
        type=pathlib.Path,

    )

    parser.add_argument(
        "--log_dir",
        help="Path where to create experiment log dirs",
        default="experiment_log/",
        type=pathlib.Path,
        # type=str
    )

    parser.add_argument(
        "--exp_name",
        help="Flag to set the name of the experiment",
        type=str
    )

    parser.add_argument(
        "--feature_extraction_method",
        help="Name of folder containing respective extracted features. Has to match {feature_extraction_method} in --root_dir argument.",
        required=False,
        type=pathlib.Path,
        default= "cls_features_non_averaged"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to use",
        choices=["AudioSetZSL", "VGGSound", "UCF", "ActivityNet"],
        default="VGGSound"
    )


    parser.add_argument(
        "--zero_shot_split",
        help="Name of zero shot split to use.",
        choices=["", "cls_split", "main_split"],
        default="cls_split"
    )
    parser.add_argument(
        "--reg_loss",
        help="Flag for setting the regularization loss",
        type=str_to_bool, nargs='?', const=True

    )
    parser.add_argument(
        "--retrain_all",
        help="Retrain with all data from train and validation",
        type=str_to_bool, nargs='?', const=True
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs",
        type=int
    )
    parser.add_argument(
        "--batch_seqlen_train",
        type=str,
        choices=["max", "fixed"]
    )
    parser.add_argument(
        "--batch_seqlen_train_maxlen",
        type=int
    )
    parser.add_argument(
        "--batch_seqlen_train_trim",
        type=str,
        choices=["random", "center"]
    )
    parser.add_argument(
        "--batch_seqlen_test",
        type=str,
        choices=["max", "fixed"]
    )
    parser.add_argument(
        "--batch_seqlen_test_maxlen",
        type=int
    )
    parser.add_argument(
        "--batch_seqlen_test_trim",
        type=str,
        choices=["random", "center"]
    )

    parser.add_argument(
        "--cross_entropy_loss",
        help="Use the crossentropy loss",
        type=str_to_bool, nargs='?', const=True
    )

    parser.add_argument(
        "--optimizer",
        help="Select Optimizer used for training",
        type=str,
        choices=["adam", "adam-sam"]
    )
    parser.add_argument(
        "--lr",
        help="Learning rate",
        type=float
    )
    parser.add_argument(
        "--bs",
        help="Batch size",
        type=int
    )
    parser.add_argument(
        "--n_batches",
        help="Number of batches for the balanced batch sampler",
        type=int
    )






    parser.add_argument(
        "--distance_fn",
        help="Distance function for the contrastive loss calculation",
        choices=["L2Loss", "SquaredL2Loss"],
        type=str
    )
    parser.add_argument(
        "--lr_scheduler",
        help="Use LR_scheduler",
        type=str,
        default="",
    )

    # defaults
    parser.add_argument(
        "--seed",
        help="Random seed",
        type=int
    )

    parser.add_argument(
        "--device",
        help="Device to run on.",
        choices=["cuda", "cpu", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    )


    parser.add_argument(
        "--rec_loss",
        type=str_to_bool, nargs='?', const=True
    )


    eval_group = parser.add_argument_group('eval')
    eval_group.add_argument(
        "--load_path_stage_A",
        help="Path to experiment log folder of stage A",
        # required=True,
        type=pathlib.Path,
        # type=str
    )
    eval_group.add_argument('--num_workers', type=int, default=0)
    eval_group.add_argument(
        "--load_path_stage_B",
        help="Path to experiment log folder of stage B",
        # required=True,
        type=pathlib.Path,
        # type=str
    )
    eval_group.add_argument(
        "--eval_name",
        help="Evaluation name to be displayed in the final output string",
        type=str,
        # required=True
    )
    eval_group.add_argument(
        "--eval_modality",
        help="Wether to evaluate performance of audio features, video features, or both combined",
        choices=["video", "audio", "both"],
        default='video',
        type=str,
        # required=True
    )
    eval_group.add_argument(
        "--eval_bs",
        help="Batch size",
        type=int
    )
    args = parser.parse_args(*args, **kwargs)


    arg_groups={}
    for group in parser._action_groups:
        group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        arg_groups[group.title]=configargparse.Namespace(**group_dict)


    eval_args = arg_groups['eval']
    shared_args_list = ['root_dir', 'dataset_name', 'device', 'batch_seqlen_test', 'batch_seqlen_test_maxlen', 'batch_seqlen_test_trim']
    shared_args_dict = {a:getattr(args,a,None) for a in shared_args_list}

    eval_main_args = configargparse.Namespace(**shared_args_dict, **vars(eval_args),)


    return args, eval_main_args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
