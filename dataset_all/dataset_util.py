from torch.utils import data
import logging
import numpy as np
import torch
from src.utils import SamplerFactory
from .VGGSound_ZSL import VGGSoundDataset
from .UCF_GZSL import UCFDataset
from .ActivityNet_GZSL import ActivityNetDataset

class ContrastiveDataset(data.Dataset):
    def __init__(self, ori_dataset):
        super(ContrastiveDataset, self).__init__()
        self.logger = logging.getLogger()
        self.logger.info(
            f"Initializing {self.__class__.__name__}\t"
            f"Based on : {ori_dataset.__class__.__name__}\t"
            f"with split: {ori_dataset.dataset_split}")
        self.ori_dataset = ori_dataset
        self.dataset_split = self.ori_dataset.dataset_split
        self.current_seen_unseen_class_ids = self.ori_dataset.current_seen_unseen_class_ids
        if self.dataset_split == "train" or self.dataset_split == "train_val":
            self.targets = self.ori_dataset.all_data['target'] #bs
            self.data = self.ori_dataset.all_data
            self.targets_set = set(self.targets.tolist()) #138
            self.target_to_indices = {target: np.where(self.targets == target)[0] 
                                      for target in self.targets_set}

        elif self.dataset_split == "val" or self.dataset_split == "test":
            self.targets = self.ori_dataset.all_data['target'] #bs
            self.data = self.ori_dataset.all_data
            self.targets_set = set(self.targets.tolist())
            self.target_to_indices = {target: np.where(self.targets == target)[0]
                                      for target in self.targets_set}

            random_state = np.random.RandomState(29)
            pos_neg_pairs = [[i, 
                              random_state.choice(self.target_to_indices[
                                                      np.random.choice( # select one random target ...
                                                          list(self.targets_set - set([self.targets[i].item()])) # ... from set of targets minus targets from current i
                                                      )
                                                  ])
                              ]
                             for i in range(len(self.targets))]
            self.val_pairs = pos_neg_pairs 
        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")

    def __len__(self):
        classes_mask = np.where(np.isin(self.targets, self.current_seen_unseen_class_ids))[0]
        return len(self.ori_dataset.all_data['target'][classes_mask])

    def __getitem__(self, index):
        if self.dataset_split == "train" or self.dataset_split == "train_val":
            positive_target = self.targets[index].item() 
            pos_target_index = list(self.targets_set).index(positive_target) 
            x_a1 = self.data["audio"][index]
            x_v1 = self.data["video"][index]
            x_t1 = self.data["text"][pos_target_index]
            x_single_textual_cloud_embeddings=self.data['textual_cloud_embeddings'][pos_target_index]

            positive_index = index
            while positive_index == index:
                # randomly select an index with positive target (until we get index that is not our current instance)
                positive_index = np.random.choice(self.target_to_indices[positive_target]) 
            negative_target = np.random.choice(list(self.targets_set - set([positive_target]))) 
            negative_index = np.random.choice(self.target_to_indices[negative_target]) 
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][negative_index]
            x_v2 = self.data["video"][negative_index]
            x_t2 = self.data["text"][neg_target_index]

        elif self.dataset_split == "val" or self.dataset_split == "test":
            positive_target = self.targets[self.val_pairs[index][0]].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][self.val_pairs[index][0]]
            x_v1 = self.data["video"][self.val_pairs[index][0]]
            x_t1 = self.data["text"][pos_target_index]
            x_single_textual_cloud_embeddings = self.data['textual_cloud_embeddings'][pos_target_index]

            negative_target = self.targets[self.val_pairs[index][1]].item() 
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][self.val_pairs[index][1]]
            x_v2 = self.data["video"][self.val_pairs[index][1]]
            x_t2 = self.data["text"][neg_target_index]

        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")

        data = {
            "positive": {"audio": x_a1, "video": x_v1, "text": x_t1, "textual_cloud_embeddings":x_single_textual_cloud_embeddings},
            "negative": {"audio": x_a2, "video": x_v2, "text": x_t2, }
        }
        target = {
            "positive": positive_target,
            "negative": negative_target
        }
        return data, target



class DefaultCollator(object):
    def __init__(self, mode='max', max_len=60, trim='random', rate_video=16/25, rate_audio=0.96):
        self.mode = mode 
        self.max_len = max_len
        self.rate_video = rate_video
        self.rate_audio = rate_audio
        self.trim = trim 

    def get_max_seq_len(self, data):
        maxlen_audio_pos=0
        maxlen_audio_neg=0
        maxlen_video_pos=0
        maxlen_video_neg=0

        # get max sequence length in batch
        for element in data:
            positive=element['positive']
            negative=element['negative']
            positive_audio_size = 1 if positive['audio'].ndim == 1 else positive['audio'].shape[0]
            positive_video_size = 1 if positive['video'].ndim == 1 else positive['video'].shape[0]
            negative_audio_size = 1 if negative['audio'].ndim == 1 else negative['audio'].shape[0]
            negative_video_size = 1 if negative['video'].ndim == 1 else negative['video'].shape[0]
            if positive_audio_size > maxlen_audio_pos:
                maxlen_audio_pos=positive_audio_size
            if negative_audio_size > maxlen_audio_neg:
                maxlen_audio_neg=negative_audio_size
            if positive_video_size > maxlen_video_pos:
                maxlen_video_pos=positive_video_size
            if negative_video_size > maxlen_video_neg:
                maxlen_video_neg=negative_video_size


        return maxlen_audio_pos, maxlen_video_pos, maxlen_audio_neg, maxlen_video_neg

    def __call__(self, batch): 
        data=[el[0] for el in batch]
        target=[el[1] for el in batch]
        batch_size=len(batch)



        if self.mode == 'max':
            len_audio_pos, len_video_pos, len_audio_neg, len_video_neg = self.get_max_seq_len(data)
        elif self.mode == 'fixed':
            # takes video features as anchor as more features per second
            len_video_pos, len_video_neg = self.max_len, self.max_len #60
            len_audio_pos, len_audio_neg = round(self.max_len *  self.rate_video * self.rate_audio), round(self.max_len *  self.rate_video * self.rate_audio)


        # init padding mask and timestep arrays
        mask_audio_pos = torch.ones((batch_size, len_audio_pos))
        mask_audio_neg=torch.ones((batch_size, len_audio_neg))
        mask_video_pos=torch.ones((batch_size, len_video_pos))
        mask_video_neg=torch.ones((batch_size, len_video_neg))

        timestep_audio_pos = np.repeat(np.expand_dims(np.arange(0, len_audio_pos, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_audio # audio has 0.96 features per second due to vggish
        timestep_audio_neg = np.repeat(np.expand_dims(np.arange(0, len_audio_neg, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_audio
        timestep_video_pos = np.repeat(np.expand_dims(np.arange(0, len_video_pos, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_video # video has 0.64 features per second
        timestep_video_neg = np.repeat(np.expand_dims(np.arange(0, len_video_neg, step=1), axis=0), batch_size, axis=0).astype(float) * self.rate_video



        for idx in range(len(data)):
            positive=data[idx]['positive']
            negative=data[idx]['negative']

            positive_audio_size = 1 if positive['audio'].ndim == 1 else positive['audio'].shape[0]
            positive_video_size = 1 if positive['video'].ndim == 1 else positive['video'].shape[0]
            negative_audio_size = 1 if negative['audio'].ndim == 1 else negative['audio'].shape[0]
            negative_video_size = 1 if negative['video'].ndim == 1 else negative['video'].shape[0]


            diff_pos_audio = positive_audio_size - len_audio_pos 
            diff_neg_audio = negative_audio_size - len_audio_neg
            diff_pos_video = positive_video_size - len_video_pos
            diff_neg_video = negative_video_size - len_video_neg

            trim_start_video_pos = None
            trim_start_video_neg = None
            trim_start_audio_pos = None
            trim_start_audio_neg = None

            if diff_pos_video < 0: 
                # padding
                mask_video_pos[idx, diff_pos_video:] = 0
                positive['video'] = np.pad(positive['video'], [(0, -diff_pos_video), (0, 0)])
            elif diff_pos_video > 0: 
                if self.trim == 'random':
                    trim_start_video_pos = torch.randint(diff_pos_video, (1,))
                elif self.trim == 'center':
                    trim_start_video_pos = torch.tensor(diff_pos_video // 2)
                trim_start_audio_pos = min(diff_pos_audio, (trim_start_video_pos * self.rate_video / self.rate_audio).round().int()) if trim_start_video_pos > 0 else 0
                # trimming
                positive['video'] = positive['video'][trim_start_video_pos:trim_start_video_pos+len_video_pos]
            else:
                pass

            if diff_pos_audio < 0:
                # padding
                mask_audio_pos[idx, diff_pos_audio:] = 0
                positive['audio'] = np.pad(positive['audio'], [(0, -diff_pos_audio), (0,0)])
            elif diff_pos_audio > 0:
                if trim_start_audio_pos == None:
                    trim_start_audio_pos = 0
                # # trimming
                # trim_start_audio_pos = torch.randint(diff_pos_audio, (1,))
                # trim_start_video_pos = round(trim_start_audio_pos * self.rate_audio / self.rate_video)
                positive['audio'] = positive['audio'][trim_start_audio_pos:trim_start_audio_pos+len_audio_pos]
            else:
                pass


            if diff_neg_video < 0:
                # padding
                mask_video_neg[idx, diff_neg_video:] = 0
                negative['video'] = np.pad(negative['video'], [(0, -diff_neg_video), (0, 0)])
            elif diff_neg_video > 0:
                if self.trim == 'random':
                    trim_start_video_neg = torch.randint(diff_neg_video, (1,))
                elif self.trim == 'center':
                    trim_start_video_neg = torch.tensor(diff_neg_video // 2)
                trim_start_audio_neg = min(diff_neg_audio, (trim_start_video_neg * self.rate_video / self.rate_audio).round().int()) if trim_start_video_neg > 0 else 0
                negative['video'] = negative['video'][trim_start_video_neg:trim_start_video_neg+len_video_neg]
                # trimming
            else:
                pass


            if diff_neg_audio < 0:
                # padding
                mask_audio_neg[idx, diff_neg_audio:] = 0
                negative['audio'] = np.pad(negative['audio'], [(0, -diff_neg_audio), (0, 0)])
            elif diff_neg_audio > 0:
                if trim_start_audio_neg == None:
                    trim_start_audio_neg = 0
                # trim_start_audio_neg = torch.randint(diff_neg_audio, (1,))
                # trim_start_video_neg = round(trim_start_audio_neg * self.rate_audio / self.rate_video)
                negative['audio'] =  negative['audio'][trim_start_audio_neg:trim_start_audio_neg+len_audio_neg]
                # trimming
            else:
                pass

            data[idx]['positive']=positive
            data[idx]['negative']=negative

        data_final={}
        target_final={}

        data_final['positive']={}
        data_final['positive']['audio']=torch.tensor([element['positive']['audio'] for element in data], dtype=torch.float32)
        data_final['positive']['video']=torch.tensor([element['positive']['video'] for element in data])
        data_final['positive']['text']=torch.tensor([element['positive']['text'] for element in data])
        # crshi=[element['positive']['text'] for element in data]

        data_final['negative']={}
        data_final['negative']['audio'] = torch.tensor([element['negative']['audio'] for element in data], dtype=torch.float32)
        data_final['negative']['video'] = torch.tensor([element['negative']['video'] for element in data])
        data_final['negative']['text']=torch.tensor([element['negative']['text'] for element in data])

        target_final['positive']=torch.tensor([element['positive'] for element in target])
        target_final['negative']=torch.tensor([element['negative'] for element in target])

        audio_textual_cloud_embedding=torch.stack([element['positive']['textual_cloud_embeddings']['audio'] for element in data]) #[bs,50,1024]
        visual_textual_cloud_embedding=torch.stack([element['positive']['textual_cloud_embeddings']['visual'] for element in data])
        context_textual_cloud_embedding = torch.stack([element['positive']['textual_cloud_embeddings']['context'] for element in data])
        data_final['positive']['audio_textual_cloud_embedding']=audio_textual_cloud_embedding
        data_final['positive']['visual_textual_cloud_embedding']=visual_textual_cloud_embedding
        data_final['positive']['context_textual_cloud_embedding']=context_textual_cloud_embedding



        return data_final, target_final





def get_dataset(args):
    if args.dataset_name == "VGGSound":
        if args.retrain_all==False:
            train_dataset = VGGSoundDataset( args=args,dataset_split="train")
        if args.retrain_all==True:
            train_val_dataset = VGGSoundDataset(args=args,dataset_split="train_val")
        val_dataset = VGGSoundDataset(args=args,dataset_split="val")

    elif args.dataset_name == "UCF":
        if args.retrain_all==False:
            train_dataset = UCFDataset(args=args,dataset_split="train")
        if args.retrain_all==True:
            train_val_dataset = UCFDataset(args=args,dataset_split="train_val",)
        val_dataset = UCFDataset(args=args,dataset_split="val",)

    elif args.dataset_name == "ActivityNet":
        if args.retrain_all==False:
            train_dataset = ActivityNetDataset(args=args,dataset_split="train",)
        if args.retrain_all==True:
            train_val_dataset = ActivityNetDataset(args=args,dataset_split="train_val")
        val_dataset = ActivityNetDataset(args=args,dataset_split="val",)

    else:
        raise NotImplementedError()

    if args.retrain_all==True:
        return train_val_dataset, val_dataset
    if args.retrain_all==False:
        return train_dataset, val_dataset

def reconstruct_dataset(train_dataset,val_all_dataset):
    
    contrastive_train_dataset=ContrastiveDataset(train_dataset)
    contrastive_val_all_dataset = ContrastiveDataset(val_all_dataset)
    return contrastive_train_dataset, contrastive_val_all_dataset

def Sampler_dataset(train_dataset,val_all_dataset,logger,args):

    train_sampler = SamplerFactory(logger).get(
            class_idxs=list(train_dataset.target_to_indices.values()),

            batch_size=args.bs,
            n_batches=args.n_batches,
            alpha=1,
            kind='random'
        )

    val_all_sampler = SamplerFactory(logger).get(
        class_idxs=list(val_all_dataset.target_to_indices.values()),
        batch_size=args.bs,
        n_batches=args.n_batches,
        alpha=1,
        kind='random'
    )
    return train_sampler, val_all_sampler