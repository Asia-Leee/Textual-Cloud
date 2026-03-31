import logging
from tqdm import tqdm
import torch
from collections import defaultdict
from .trian_util import check_best_loss, check_best_score, save_best_model
from .trian_util import add_logs_tensorboard,add_loss_details
import torch.nn.functional as F


def train(train_loader, val_loader, model, epochs, device, metric, log_dir, args):
    best_loss = None
    best_score = None
    best_hm_epoch = None
    best_loss_epoch = None

    visual_losses_train = defaultdict(list)
    visual_losses_val = defaultdict(list)
    visual_hm_epoch = []
    visual_lr_epochs=[]

    for epoch in range(epochs):
        
        visual_train_loss_epoch,visual_lr_epoch=train_step(train_loader, model,epoch,epochs, device, metric, args)
        
        _, val_hm,visual_val_loss_epoch = val_step(val_loader, model, epoch, epochs,  device, metric, args)
        
        best_score, best_hm_epoch = check_best_score(epoch, best_score, best_hm_epoch, val_hm, model,log_dir)
        
        model.optimize_scheduler(val_hm)

        for k ,v in visual_train_loss_epoch.items(): visual_losses_train[k].append(v)
        for k ,v in visual_val_loss_epoch.items(): visual_losses_val[k].append(v)
        visual_hm_epoch.append(val_hm)
        visual_lr_epochs.append(visual_lr_epoch)

    if args.best_model_criterion == 'loss': 
        return best_loss, best_score, best_loss_epoch
    elif args.best_model_criterion == 'score':
        return best_loss, best_score, best_hm_epoch, visual_losses_train, visual_losses_val,visual_hm_epoch,visual_lr_epochs


def train_step(data_loader, model, epoch,epochs, device, metric,  args):
    logger = logging.getLogger()
    model.train()

    metric.reset()

    text_embeddings_all_current, mapping_dict,text_embeddings_all_current_new,text_embeddings_all_current_list = data_loader.dataset.ori_dataset.map_embeddings_target

    batch_loss_details={}
    visual_loss_epoch=defaultdict(list)
    visual_lr_epoch=[]
    for batch_idx, (data, target) in tqdm(enumerate(data_loader),mininterval=args.mininterval):
        p = data["positive"]
        q = data["negative"] 

        p_audio_embeddings = p["audio"].to(device)
        p_audio_embeddings = p_audio_embeddings+torch.randn_like(p_audio_embeddings) * args.aug_noise_level
        p_video_embeddings = p["video"].to(device)
        p_video_embeddings = p_video_embeddings + torch.randn_like(p_video_embeddings) * args.aug_noise_level
        p_text_embeddings = p["text"].to(device)
        p_targets = target["positive"].to(device)

        
        p_textual_cloud_audio_embeddings=p['audio_textual_cloud_embedding'].to(device) #[bs,50,1024]
        p_textual_cloud_visual_embeddings=p['visual_textual_cloud_embedding'].to(device)
        p_textual_cloud_context_embeddings=p['context_textual_cloud_embedding'].to(device)

        
        p_textual_audio_embedding=torch.mean(p_textual_cloud_audio_embeddings,dim=1) #[bs,1024]
        p_textual_visual_embedding=torch.mean(p_textual_cloud_visual_embeddings,dim=1)
        p_textual_context_embedding=torch.mean(p_textual_cloud_context_embeddings,dim=1)

        p_textual_audio_embedding=F.normalize(p_textual_audio_embedding,dim=1,p=2)
        p_textual_visual_embedding=F.normalize(p_textual_visual_embedding,dim=1,p=2)
        p_textual_context_embedding=F.normalize(p_textual_context_embedding,dim=1,p=2)

        
        pass


        p_new_text_embeddings = torch.cat((p_textual_visual_embedding,p_textual_audio_embedding),dim=1)


        
        p_targets_index=torch.tensor([mapping_dict[int(i)] for i in p_targets],device=device)

        loss_total, loss_details,visual_lr = model.optimize_params( #forward+loss
            audio_embeddings=p_audio_embeddings,
            video_embeddings=p_video_embeddings,
            targets=p_targets_index,
            text_embeddings=p_text_embeddings,
            textual_audio_embedding=p_textual_audio_embedding,
            textual_visual_embedding=p_textual_visual_embedding,
            textual_context_embedding=p_textual_context_embedding,
            text_embeddings_all_current=text_embeddings_all_current_list,
            args=args,
            optimize=True,
        )

        batch_loss_details=add_loss_details(loss_details, batch_loss_details)
        for k,v in loss_details.items(): visual_loss_epoch[k].append(v.item())
        visual_lr_epoch.append(visual_lr)

        iteration = len(data_loader) * epoch + batch_idx

    
    average_loss_details = {k: v / (batch_idx+1) for k, v in batch_loss_details.items()}
    loss_msg = " | ".join([f"{k}: {v.item():.4f}" for k, v in average_loss_details.items()])
    logger.info(
        f"TRAIN\t"
        f"Epoch: {epoch}/{epochs}\t"
        f"Iteration: {iteration}\t"
        f"{loss_msg}"
    )
    return visual_loss_epoch,visual_lr_epoch


def val_step(data_loader, model,  epoch, epochs, device, metric, args=None):

    logger = logging.getLogger()
    model.eval()

    metric.reset()

    text_embeddings_all_current, mapping_dict ,text_embeddings_all_current_new,text_embeddings_all_current_list = data_loader.dataset.ori_dataset.map_embeddings_target
    with torch.no_grad():
        hm_score = 0
        seen_score=0
        unseen_score=0
        batch_loss_details={}
        visual_loss_epoch = defaultdict(list)
        for batch_idx, (data, target) in tqdm(enumerate(data_loader),mininterval=args.mininterval,desc=""):
            p = data["positive"]
            q = data["negative"]

            p_audio_embeddings = p["audio"].to(device)
            p_video_embeddings = p["video"].to(device)
            p_text_embeddings = p["text"].to(device)
            p_targets = target["positive"].to(device)

            p_targets_index = torch.tensor([mapping_dict[int(i)] for i in p_targets], device=device)


            
            p_textual_cloud_audio_embeddings = p['audio_textual_cloud_embedding'].to(device)  # [bs,50,1024]
            p_textual_cloud_visual_embeddings = p['visual_textual_cloud_embedding'].to(device)
            p_textual_cloud_context_embeddings = p['context_textual_cloud_embedding'].to(device)

            
            p_textual_audio_embedding = torch.mean(p_textual_cloud_audio_embeddings, dim=1)  # [bs,1024]
            p_textual_visual_embedding = torch.mean(p_textual_cloud_visual_embeddings, dim=1)
            p_textual_context_embedding = torch.mean(p_textual_cloud_context_embeddings, dim=1)
                
            p_textual_audio_embedding = F.normalize(p_textual_audio_embedding, dim=1, p=2)
            p_textual_visual_embedding = F.normalize(p_textual_visual_embedding, dim=1, p=2)
            p_textual_context_embedding = F.normalize(p_textual_context_embedding, dim=1, p=2)

            
            pass

            p_new_text_embeddings = torch.cat((p_textual_visual_embedding, p_textual_audio_embedding), dim=1)


            loss, loss_details,_ = model.optimize_params( 
                audio_embeddings=p_audio_embeddings,
                video_embeddings=p_video_embeddings,
                targets=p_targets_index,
                text_embeddings=p_text_embeddings,
                textual_audio_embedding=p_textual_audio_embedding,
                textual_visual_embedding=p_textual_visual_embedding,
                textual_context_embedding=p_textual_context_embedding,
                text_embeddings_all_current=text_embeddings_all_current_list,
                args=args,
                optimize=False)

            batch_loss_details = add_loss_details(loss_details, batch_loss_details)
            for k, v in loss_details.items(): visual_loss_epoch[k].append(v.item())
            iteration = len(data_loader) * epoch + batch_idx

        
        metric()
        values=metric.value()
        hm_score = values.get("both_hm", None)
        zsl_score = values.get("both_zsl", None)
        seen_score = values.get("both_seen", None)
        unseen_score = values.get("both_unseen", None)

        average_loss_details = {k: v / (batch_idx + 1) for k, v in batch_loss_details.items()}
        # loss_msg = " | ".join([f"{k}: {v.item():.4f}" for k, v in average_loss_details.items()])

        logger.info(
            f"VALID\t"
            f"Epoch: {epoch}/{epochs}\t"
            f"Iteration: {iteration}\t"
            f"ZSL: {zsl_score:.4f}\t"
            f"Seen: {seen_score:.4f}\t"
            f"Unseen: {unseen_score:.4f}\t"
            f"HM: {hm_score:.4f}\t"
            f"total_loss: {average_loss_details['total_loss']:.4f}"
        )
    return average_loss_details['total_loss'], hm_score,visual_loss_epoch
