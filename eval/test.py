import logging
import torch
from tqdm import tqdm
import numpy as np


def test( val_dataset, val_loader,test_dataset, test_loader,model_A, model_B, device, distance_fn,args):
    logger = logging.getLogger()
    model_A.eval()
    model_B.eval()

    
    val_evaluation = evaluate_dataset_baseline(val_dataset, val_loader,model_A, device, distance_fn,args=args,)

    best_beta_combined = 1. / 3 * (
            val_evaluation['audio']['beta'] + val_evaluation['video']['beta'] + val_evaluation['both']['beta'] + 1e-10)
    logger.info(
        f"Validation betas:\tAudio={val_evaluation['audio']['beta']}\tVideo={val_evaluation['video']['beta']}\tBoth={val_evaluation['both']['beta']}")
    logger.info(f"Best beta combined: {best_beta_combined}")


    
    test_evaluation = evaluate_dataset_baseline(test_dataset,test_loader, model_B, device, distance_fn,
                                                best_beta=best_beta_combined,
                                                args=args,)

    if args.dataset_name == "AudioSetZSL":
        output_string = fr"""
                   Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Unseen performance={100 * test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100 * test_evaluation["both"]["hm"]:.2f}, ZSL performance={100 * test_evaluation["both"]["zsl"]:.2f}
                   """
    elif args.dataset_name == "VGGSound" or args.dataset_name == "UCF" or args.dataset_name == "ActivityNet":
        output_string = fr"""
                    Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Unseen performance={100 * test_evaluation["both"]["unseen"]:.2f}, GZSL performance={100 * test_evaluation["both"]["hm"]:.2f}, ZSL performance={100 * test_evaluation["both"]["zsl"]:.2f}
                    """
    else:
        raise NotImplementedError()

    logger.info(output_string)


def evaluate_dataset_baseline(dataset,dataloader, model, device, distance_fn, best_beta=None,args=None,):

    dataset=dataset
    data_loader=dataloader
    text_embeddings_all_current = torch.tensor(dataset.all_data['text']).to(device) 
    _,_,_,text_embeddings_all_list=dataset.map_embeddings_target 

    accumulated_audio_emb=[]
    accumulated_video_emb=[]
    accumulated_both_emb=[]
    accumulated_targets=[]

    for batch_idx, (data, target) in tqdm(enumerate(data_loader),mininterval=args.mininterval,desc='stacking embeddings'):
        data_audio = data['positive']["audio"].to(device) #[bs,seqence_length]
        data_video = data['positive']["video"].to(device)
        targets = target["positive"].to(device) # [bs,]

        model.eval()
        with torch.no_grad():
            audio_emb, video_emb,video_audio_emb, text_emb_all_current = model.get_embeddings(data_audio, data_video,text_embeddings_all_list[0],text_embeddings_all_list[1],text_embeddings_all_list[2],)
            accumulated_audio_emb.append(audio_emb)
            accumulated_video_emb.append(video_emb)
            accumulated_both_emb.append(video_audio_emb)
        accumulated_targets.append(targets)

    stacked_audio_emb=torch.cat(accumulated_audio_emb) 
    stacked_video_emb=torch.cat(accumulated_video_emb)
    stacked_both_emb=torch.cat(accumulated_both_emb)
    stacked_targets=torch.cat(accumulated_targets)

    video_evaluation = get_best_evaluation(dataset, stacked_targets, stacked_audio_emb, stacked_video_emb, stacked_both_emb,text_emb_all_current,
                                        device=device,distance_fn=distance_fn, best_beta=best_beta, args=args)
    return {
        "audio": video_evaluation,
        "video": video_evaluation,
        "both": video_evaluation
    }


def get_best_evaluation(dataset, targets, audio_emb, video_emb, video_audio_emb,text_emb, device, distance_fn, best_beta=None, args=None):
    seen_scores = []
    zsl_scores = []
    unseen_scores = []
    hm_scores = []
    per_class_recalls = []


    start,end = 0,5
    steps = (end - start) * 15 + 1
    betas = torch.tensor([best_beta], dtype=torch.float, device=device) if best_beta else torch.linspace(start, end, steps,
                                                                                                         device=device)
    seen_label_array = torch.tensor(dataset.current_seen_class_ids, dtype=torch.long, device=device)
    unseen_label_array = torch.tensor(dataset.current_unseen_class_ids, dtype=torch.long, device=device)
    seen_unseen_array = torch.tensor(np.sort(np.concatenate((dataset.current_seen_class_ids, dataset.current_unseen_class_ids))),
                                     dtype=torch.long, device=device)

    with torch.no_grad():
        for beta in betas:
            
            distance_mat_GZSL = torch.zeros((video_emb.shape[0], len(dataset.all_class_idx)), dtype=torch.float,
                                       device=device) + 99999999999999
            distance_mat_ZSL = torch.zeros((video_emb.shape[0], len(dataset.all_class_idx)), dtype=torch.float,
                                           device=device) + 99999999999999
            if args.xiaorong_textual_cloud == "audio":
                audio_distance = torch.cdist(audio_emb, text_emb['textual_audio'], p=2)
                if distance_fn == "SquaredL2Loss":
                    audio_distance = audio_distance.pow(2)
                distance_mat_GZSL[:, seen_unseen_array] = audio_distance

                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array] = 99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL
            elif args.xiaorong_textual_cloud == "visual":
                video_distance = torch.cdist(video_emb, text_emb['textual_visual'], p=2)
                if distance_fn == "SquaredL2Loss":
                    video_distance = video_distance.pow(2)
                distance_mat_GZSL[:, seen_unseen_array] = video_distance
                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array] = 99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL
            elif args.xiaorong_textual_cloud == "visual+audio":
                video_distance = torch.cdist(video_emb, text_emb['textual_visual'], p=2)
                audio_distance = torch.cdist(audio_emb, text_emb['textual_audio'], p=2)
                distance_mat_GZSL[:, seen_unseen_array] = (video_distance+audio_distance)
                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array] = 99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL
            elif args.xiaorong_textual_cloud == "visual+context":
                video_distance = torch.cdist(video_emb, text_emb['textual_visual'], p=2)
                context_distance = torch.cdist(video_audio_emb, text_emb['textual_context'], p=2)
                distance_mat_GZSL[:, seen_unseen_array] = (video_distance+context_distance)
                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array] = 99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL
            elif args.xiaorong_textual_cloud == "audio+context":
                audio_distance = torch.cdist(audio_emb, text_emb['textual_audio'], p=2)
                context_distance = torch.cdist(video_audio_emb, text_emb['textual_context'], p=2)
                distance_mat_GZSL[:, seen_unseen_array] = (audio_distance+context_distance)
                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array] = 99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL
            elif args.xiaorong_textual_cloud == "context":
                context_distance = torch.cdist(video_audio_emb, text_emb['textual_context'], p=2)
                distance_mat_GZSL[:, seen_unseen_array] = context_distance
                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array] = 99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL

            elif args.xiaorong_textual_cloud == "both":
                # L2
                audio_distance = torch.cdist(audio_emb, text_emb['textual_audio'], p=2)
                video_distance = torch.cdist(video_emb, text_emb['textual_visual'], p=2)
                context_distance = torch.cdist(video_audio_emb, text_emb['textual_context'], p=2)
                if distance_fn == "SquaredL2Loss":
                    audio_distance = audio_distance.pow(2)
                    video_distance = video_distance.pow(2)
                distance_mat_GZSL[:, seen_unseen_array] = (audio_distance + video_distance + context_distance) 

                mask_ZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device)
                mask_ZSL[seen_label_array]=99999999999999
                distance_mat_ZSL = distance_mat_GZSL + mask_ZSL

            
            mask_GZSL = torch.zeros(len(dataset.all_class_idx), dtype=torch.long, device=device) + beta
            mask_GZSL[unseen_label_array] = 0  
            neighbor_batch = torch.argmin(distance_mat_GZSL + mask_GZSL, dim=1) 

            match_idx = neighbor_batch.eq(targets.int()).nonzero().flatten() 
            match_counts = torch.bincount(neighbor_batch[match_idx], minlength=len(dataset.all_class_idx))[
                seen_unseen_array] 
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_idx))[seen_unseen_array] 

            per_class_recall = torch.zeros(len(dataset.all_class_idx), dtype=torch.float, device=device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts 
            seen_recall_dict = per_class_recall[seen_label_array]  
            unseen_recall_dict = per_class_recall[unseen_label_array] 

            s = seen_recall_dict.mean() 
            u = unseen_recall_dict.mean()
            hm = (2 * u * s) / ((u + s) + np.finfo(float).eps)

            
            neighbor_batch_zsl = torch.argmin(distance_mat_ZSL, dim=1)
            match_idx = neighbor_batch_zsl.eq(targets.int()).nonzero().flatten()
            match_counts = torch.bincount(neighbor_batch_zsl[match_idx], minlength=len(dataset.all_class_idx))[
                seen_unseen_array]
            target_counts = torch.bincount(targets, minlength=len(dataset.all_class_idx))[seen_unseen_array]
            per_class_recall = torch.zeros(len(dataset.all_class_idx), dtype=torch.float, device=device)
            per_class_recall[seen_unseen_array] = match_counts / target_counts
            zsl = per_class_recall[unseen_label_array].mean()

            zsl_scores.append(zsl.item())
            seen_scores.append(s.item())
            unseen_scores.append(u.item())
            hm_scores.append(hm.item())
            per_class_recalls.append(per_class_recall.tolist()) 
        argmax_hm = np.argmax(hm_scores) 
        max_seen = seen_scores[argmax_hm]
        max_zsl = zsl_scores[argmax_hm]
        max_unseen = unseen_scores[argmax_hm]
        max_hm = hm_scores[argmax_hm]
        max_recall = per_class_recalls[argmax_hm]  
        best_beta = betas[argmax_hm].item() 
    return {
        "seen": max_seen,
        "unseen": max_unseen,
        "hm": max_hm,
        "recall": max_recall,
        "zsl": max_zsl,
        "beta": best_beta
    }

