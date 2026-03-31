import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils import data
from src.utils import  get_class_names

class ActivityNetDataset(data.Dataset):
    def __init__(self, args, dataset_split,):
        super(ActivityNetDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing Dataset {self.__class__.__name__}\t"
            f"split: {dataset_split}\t")
        self.args = args
        self.root = args.data_dir
        self.dataset_name = args.dataset_name
        self.feature_extraction_method = args.feature_extraction_method
        self.dataset_split = dataset_split
        self.zero_shot_split = args.zero_shot_split

        self.check_exist()
        self.get_class_names_and_idxs()
        self.get_current_seen_and_unseen_class_names_and_ids()
        self.data_pkl = self.get_data_from_pkl()

        self.textual_cloud()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data_pkl)

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data_pkl["audio"]["target"], self.current_seen_unseen_class_ids))[0]
        return {
            "audio": np.array(self.data_pkl["audio"]["data"])[classes_mask],
            "video": np.array(self.data_pkl["video"]["data"])[classes_mask],
            "text": np.array(self.data_pkl["text"]["data"])[sorted(self.current_seen_unseen_class_ids.astype(int))],
            "textual_cloud_embeddings": np.array(self.all_textual_cloud_embeddings_list)[sorted(self.current_seen_unseen_class_ids.astype(int))],
            "target": np.array(self.data_pkl["audio"]["target"])[classes_mask],
        }


    def get_class_names_and_idxs(self):

        self.all_class_names=get_class_names(self.root / "class-split/all_class.txt")
        self.all_class_name_map_idx = {_class: i for i, _class in enumerate(sorted(self.all_class_names))}
        self.all_class_idx = np.asarray([self.all_class_name_map_idx[name] for name in self.all_class_names])

        self.train_train_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_train.txt")
        self.train_train_ids = np.asarray([self.all_class_name_map_idx[name] for name in self.train_train_class_names])


        self.val_seen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")
        self.val_seen_ids = np.asarray([self.all_class_name_map_idx[name] for name in self.val_seen_class_names])
        self.val_unseen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")
        self.val_unseen_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.val_unseen_class_names])

        self.test_train_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_train.txt")
        self.test_seen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")
        self.test_unseen_class_names=get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

        self.test_train_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.test_train_class_names])
        self.test_seen_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.test_seen_class_names])
        self.test_unseen_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.test_unseen_class_names])


    def get_current_seen_and_unseen_class_names_and_ids(self):
        if self.dataset_split == "train":
            self.current_seen_class_names=self.train_train_class_names
            self.current_unseen_class_names=np.array([])
        elif self.dataset_split == "val":
            self.current_seen_class_names = self.val_seen_class_names
            self.current_unseen_class_names = self.val_unseen_class_names
        elif self.dataset_split == "train_val":
            self.current_seen_class_names = np.concatenate((self.train_train_class_names, self.val_unseen_class_names))
            self.current_unseen_class_names = np.array([])
        elif self.dataset_split == "test":
            self.current_seen_class_names = self.test_seen_class_names
            self.current_unseen_class_names = self.test_unseen_class_names

        self.current_seen_class_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.current_seen_class_names])
        self.current_unseen_class_ids=np.asarray([self.all_class_name_map_idx[name] for name in self.current_unseen_class_names])
        self.current_seen_unseen_class_ids=np.sort(np.concatenate((self.current_seen_class_ids, self.current_unseen_class_ids)))

    def get_data_from_pkl(self):
        if self.dataset_split == "train":
            data_file = self.train_file_path
        elif self.dataset_split == "val":
            data_file = self.val_file_path
        elif self.dataset_split == "train_val":
            data_file = self.trainval_file_path
        elif self.dataset_split == "test":
            data_file = self.test_file_path

        self.logger.info(f"Loading processed data from {data_file}")
        with data_file.open('rb') as f:
            x=pickle.load(f)
            return x

    @property
    def map_embeddings_target(self):
        current_text_embedding = torch.Tensor(self.data_pkl["text"]["data"])[sorted(self.current_seen_unseen_class_ids)].to(self.args.device)

        current_textual_cloud_embeddings=np.array(self.all_textual_cloud_embeddings_list)[sorted(self.current_seen_unseen_class_ids.astype(int))]
        current_textual_cloud_embedding_audio=torch.mean(torch.stack([item['audio'] for item in current_textual_cloud_embeddings]),dim=1).to(self.args.device)
        current_textual_cloud_embedding_visual=torch.mean(torch.stack([item['visual'] for item in current_textual_cloud_embeddings]),dim=1).to(self.args.device)
        current_textual_cloud_embedding_context=torch.mean(torch.stack([item['context'] for item in current_textual_cloud_embeddings]),dim=1).to(self.args.device)

        
        current_textual_cloud_embedding_visual=F.normalize(current_textual_cloud_embedding_visual,dim=1,p=2)
        current_textual_cloud_embedding_audio=F.normalize(current_textual_cloud_embedding_audio,dim=1,p=2)
        current_textual_cloud_embedding_context=F.normalize(current_textual_cloud_embedding_context,dim=1,p=2)
        current_textual_cloud_embedding_list=[current_textual_cloud_embedding_audio,current_textual_cloud_embedding_visual,current_textual_cloud_embedding_context]

        current_text_embedding_new=torch.cat((current_textual_cloud_embedding_visual,current_textual_cloud_embedding_audio),dim=1)
        sorted_classes_ids = sorted(self.current_seen_unseen_class_ids)  # [5,12,13,...]
        mapping_dict = {}
        for i in range(len(sorted_classes_ids)):  
            mapping_dict[int(sorted_classes_ids[i])] = i
        return current_text_embedding, mapping_dict,current_text_embedding_new,current_textual_cloud_embedding_list

    def check_exist(self):
        self.train_file_path = Path(
            "/Dataset/ActivityNet/_features_processed/cls_features_non_averaged/trainingcls_split.pkl")
        self.trainval_file_path = Path(
            "/Dataset/ActivityNet/_features_processed/cls_features_non_averaged/train_valcls_split.pkl")
        self.val_file_path = Path(
            "/Dataset/ActivityNet/_features_processed/cls_features_non_averaged/valcls_split.pkl")
        self.test_file_path = Path(
            "/Dataset/ActivityNet/_features_processed/cls_features_non_averaged/testcls_split.pkl")
        if self.train_file_path.exists() and self.trainval_file_path.exists() and self.val_file_path.exists() and self.test_file_path.exists():
            return True
        else:
            raise FileNotFoundError

    def textual_cloud(self):
        textual_cloud_embedding_path = \
            "/Dataset/activitynet_descriptions_llama_(500)_top50.pt"
        ori_textual_cloud_embeddings = torch.load(textual_cloud_embedding_path, map_location='cpu')
        df=pd.read_csv("/Dataset/ActivityNet/class-split/activitynet_w2v_class_names.csv")
        mapping_dict=dict(zip(df['original'],df['manual']))

        self.all_textual_cloud_embeddings_dict={k:ori_textual_cloud_embeddings[mapping_dict[k]] for k in self.all_class_names}
        self.all_textual_cloud_embeddings_list = [ori_textual_cloud_embeddings[mapping_dict[k]] for k in self.all_class_names]

        print('done')