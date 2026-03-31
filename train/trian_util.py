import torch
import logging
def add_logs_tensorboard(batch_loss_details, writer, batch_idx, step, which_stage):


    writer.add_scalar(f"Loss/total_loss_"+which_stage, batch_loss_details['Loss/total_loss']/(batch_idx), step)
    writer.add_scalar(f"Loss/loss_reg_"+which_stage, batch_loss_details['Loss/loss_reg']/(batch_idx), step)
    writer.add_scalar(f"Loss/loss_cmd_rec_"+which_stage, batch_loss_details['Loss/loss_rec']/(batch_idx), step)
    writer.add_scalar(f"Loss/cross_entropy_"+which_stage, batch_loss_details['Loss/loss_ce']/(batch_idx), step)

def add_loss_details(current_loss_details, batch_loss_details):
    for key, value in current_loss_details.items():
        if key not in batch_loss_details:
            batch_loss_details[key]=value
        else:
            batch_loss_details[key]+=value
    return batch_loss_details


def check_best_loss(epoch, best_loss, best_epoch, val_loss, model, optimizer, log_dir, args):
    if not best_loss:
        save_best_model(epoch, val_loss, model, optimizer, log_dir, args, metric="loss")
        return val_loss, epoch

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        save_best_model(epoch, best_loss, model, optimizer, log_dir, args, metric="loss")
    return best_loss, best_epoch


def check_best_score(epoch, best_score, best_epoch, hm_score, model,  log_dir):
    if not best_score:
        save_best_model(epoch, hm_score, model, log_dir,  metric="score")
        return hm_score, epoch
    if hm_score > best_score:
        best_score = hm_score
        best_epoch = epoch
        save_best_model(epoch, best_score, model, log_dir,  metric="score")
    return best_score, best_epoch

def save_best_model(epoch, best_metric, model, log_dir, metric="",):
    logger = logging.getLogger()
    logger.info(f"Saving model to {log_dir} with {metric} = {best_metric:.4f}")
    optimizer = model.optimizer_gen
    save_dict = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metric": metric
    }
    torch.save(
        save_dict,
        log_dir / f"{model.__class__.__name__}_best_{metric}.pt"
    )