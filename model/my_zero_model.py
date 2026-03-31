import torch
import torch.nn as nn
import torch.optim as optim
# user defined
from src.optimizer import SAM
from .linear_module import EmbeddingNet
from .transformer_module import GatedTriModalInteraction


torch.set_printoptions(threshold=10_000) 
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


class My_zero_shot_model(nn.Module):
    def __init__(self, args,):
        super(My_zero_shot_model, self).__init__()

        #loss
        self.rec_loss = args.rec_loss
        self.reg_loss = args.reg_loss
        self.cross_entropy_loss = args.cross_entropy_loss

        #optimizer
        self.lr_scheduler = args.lr_scheduler

        #other
        self.modality = args.modality  # both
        self.xiaorong_textual_cloud=args.xiaorong_textual_cloud
        self.xiaorong_GGA=args.xiaorong_GGA


        #Architecture
        hidden_size=args.hidden_size
        dropout=args.dropout
        self.video_enc = EmbeddingNet(input_size=512,hidden_size=hidden_size,output_size=64,dropout=dropout,use_bn=True)
        self.audio_enc = EmbeddingNet(input_size=1024,hidden_size=hidden_size,output_size=64,dropout=dropout,use_bn=True)
        self.audio_video_enc=EmbeddingNet(input_size=1536,hidden_size=hidden_size,output_size=64,dropout=dropout,use_bn=True)

        self.textual_enc_audio = EmbeddingNet(input_size=1024, hidden_size=hidden_size,output_size=512,dropout=dropout,use_bn=True)
        self.textual_enc_visual = EmbeddingNet(input_size=512, hidden_size=hidden_size,output_size=512, dropout=dropout,use_bn=True)
        self.textual_enc_context= EmbeddingNet(input_size=512, hidden_size=hidden_size,output_size=512, dropout=dropout,use_bn=True)

        self.tri_attention= GatedTriModalInteraction(input_dim=512,nhead=8,dim_feedforward=2048,num_layers=1)

        self.textual_proj_audio = EmbeddingNet(input_size=512,hidden_size=hidden_size,output_size=64,dropout=dropout, use_bn=True)
        self.textual_proj_visual=EmbeddingNet(input_size=512,hidden_size=hidden_size,output_size=64,dropout=dropout, use_bn=True)
        self.textual_proj_context=EmbeddingNet(input_size=512, hidden_size=hidden_size,output_size=64, dropout=dropout, use_bn=True)


        # Optimizer
        self.lr = args.lr
        if args.optimizer == 'adam':
            self.optimizer_gen = optim.Adam( 
                self.parameters(),
                lr=self.lr, weight_decay=args.weight_decay
            )
        if self.lr_scheduler=="reduce":
            self.scheduler_learning_rate = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max',
                                                                                patience=3, verbose=True)
        if self.lr_scheduler == 'cosine':
            self.scheduler_learning_rate=optim.lr_scheduler.CosineAnnealingLR(self.optimizer_gen,T_max=args.epochs,eta_min=1e-6)

        self.criterion_cls = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()

        print('Done')

    def optimize_scheduler(self, value): 
        if self.lr_scheduler=='reduce':
            self.scheduler_learning_rate.step(value)
        elif self.lr_scheduler == 'cosine':
            self.scheduler_learning_rate.step()

    def forward(self, audio, video, text_embedding,textual_audio_embedding,textual_visual_embedding,textual_context_embedding):
        video = video.type(torch.float32)
        textual_visual_embedding=textual_visual_embedding.type(torch.float32)
        textual_context_embedding=textual_context_embedding.type(torch.float32)

        video_audio=torch.cat((video,audio),dim=-1)
        video=self.video_enc(video)
        audio =self.audio_enc(audio)
        video_audio=self.audio_video_enc(video_audio)

        if self.xiaorong_textual_cloud=="audio":
            textual_audio_embedding = self.textual_enc_audio(textual_audio_embedding)
            textual_audio_embedding, _, _ = self.tri_attention(textual_audio_embedding, textual_audio_embedding, textual_audio_embedding)
            textual_audio_embedding = self.textual_proj_audio(textual_audio_embedding)

        elif self.xiaorong_textual_cloud=="visual":
            textual_visual_embedding = self.textual_enc_visual(textual_visual_embedding)
            _, textual_visual_embedding, _ = self.tri_attention(textual_visual_embedding, textual_visual_embedding,textual_visual_embedding)
            textual_visual_embedding=self.textual_proj_visual(textual_visual_embedding)

        elif self.xiaorong_textual_cloud=="visual+audio":
            textual_audio_embedding = self.textual_enc_audio(textual_audio_embedding)
            textual_visual_embedding = self.textual_enc_visual(textual_visual_embedding)
            textual_audio_embedding, textual_visual_embedding, _ = self.tri_attention(textual_audio_embedding, textual_visual_embedding,textual_audio_embedding+textual_visual_embedding)
            textual_audio_embedding = self.textual_proj_audio(textual_audio_embedding)
            textual_visual_embedding = self.textual_proj_visual(textual_visual_embedding)
        elif self.xiaorong_textual_cloud == "visual+context":
            textual_visual_embedding = self.textual_enc_visual(textual_visual_embedding)
            textual_context_embedding = self.textual_enc_context(textual_context_embedding)
            _, textual_visual_embedding, textual_context_embedding = self.tri_attention(textual_visual_embedding+textual_context_embedding,
                                                                                  textual_visual_embedding,
                                                                                  textual_context_embedding)
            textual_visual_embedding=self.textual_proj_visual(textual_visual_embedding)
            textual_context_embedding=self.textual_proj_context(textual_context_embedding)
        elif self.xiaorong_textual_cloud == "audio+context":
            textual_audio_embedding = self.textual_enc_audio(textual_audio_embedding)
            textual_context_embedding = self.textual_enc_context(textual_context_embedding)
            textual_audio_embedding, _, textual_context_embedding = self.tri_attention(
                                                                                textual_audio_embedding,
                                                                                textual_audio_embedding+textual_context_embedding,
                                                                                textual_context_embedding)

            textual_audio_embedding = self.textual_proj_audio(textual_audio_embedding)
            textual_context_embedding = self.textual_proj_context(textual_context_embedding)
        elif self.xiaorong_textual_cloud=="context":
            textual_context_embedding = self.textual_enc_context(textual_context_embedding)
            _, _, textual_context_embedding = self.tri_attention(textual_context_embedding, textual_context_embedding,textual_context_embedding)
            textual_context_embedding=self.textual_proj_context(textual_context_embedding)

        elif self.xiaorong_textual_cloud=="both":
            textual_audio_embedding = self.textual_enc_audio(textual_audio_embedding)
            textual_visual_embedding = self.textual_enc_visual(textual_visual_embedding)
            textual_context_embedding = self.textual_enc_context(textual_context_embedding)
            if self.xiaorong_GGA=="both":
                textual_audio_embedding,textual_visual_embedding,textual_context_embedding=self.tri_attention(textual_audio_embedding,textual_visual_embedding, textual_context_embedding)

            textual_audio_embedding=self.textual_proj_audio(textual_audio_embedding)
            textual_visual_embedding=self.textual_proj_visual(textual_visual_embedding)
            textual_context_embedding=self.textual_proj_context(textual_context_embedding)


        outputs = {
            "textual_audio":textual_audio_embedding,
            "textual_visual":textual_visual_embedding,
            "textual_context":textual_context_embedding,
            "audio":audio,
            "video":video,
            "video_audio":video_audio,
        }

        return outputs

    def compute_loss(self, outputs, text_embeddings_all_train, target,args,optimize):

        textual_audio_embedding = outputs['textual_audio']
        textual_visual_embedding = outputs['textual_visual']
        textual_context_embedding = outputs['textual_context']
        audio = outputs['audio']
        video = outputs['video']
        video_audio = outputs['video_audio']

        #get train embedding
        if self.xiaorong_textual_cloud == "audio":
            textual_audio_all_current = self.textual_enc_audio(text_embeddings_all_train[0])
            textual_audio_all_current, _,_ = self.tri_attention(textual_audio_all_current, textual_audio_all_current, textual_audio_all_current)
            textual_audio_all_current = self.textual_proj_audio(textual_audio_all_current)

            scores_audio = torch.matmul(audio, textual_audio_all_current.t())
            l_ce_audio = self.criterion_cls(scores_audio, target)
            l_reg_audio = (self.MSE_loss(textual_audio_embedding, audio))
            l_reg_total = l_reg_audio
            l_ce_total = l_ce_audio


        elif self.xiaorong_textual_cloud == "visual":
            textual_visual_all_current=self.textual_enc_visual(text_embeddings_all_train[1].type(torch.float32))
            _, textual_visual_all_current,_ = self.tri_attention(textual_visual_all_current, textual_visual_all_current, textual_visual_all_current)
            textual_visual_all_current = self.textual_proj_visual(textual_visual_all_current)


            scores_visual = torch.matmul(video, textual_visual_all_current.t())
            l_ce_visual = self.criterion_cls(scores_visual, target)
            l_reg_visual = (self.MSE_loss(textual_visual_embedding,video))
            l_reg_total = l_reg_visual
            l_ce_total = l_ce_visual
        elif self.xiaorong_textual_cloud =="visual+audio":
            textual_audio_all_current = self.textual_enc_audio(text_embeddings_all_train[0])
            textual_visual_all_current = self.textual_enc_visual(text_embeddings_all_train[1].type(torch.float32))
            textual_audio_all_current, textual_visual_all_current, textual_context_all_current = self.tri_attention(
                textual_audio_all_current, textual_visual_all_current, textual_audio_all_current+textual_visual_all_current)
            textual_audio_all_current = self.textual_proj_audio(textual_audio_all_current)
            textual_visual_all_current = self.textual_proj_visual(textual_visual_all_current)

            scores_audio = torch.matmul(audio, textual_audio_all_current.t())
            scores_visual = torch.matmul(video, textual_visual_all_current.t())
            l_ce_audio = self.criterion_cls(scores_audio, target)
            l_reg_audio = (self.MSE_loss(textual_audio_embedding, audio))
            l_ce_visual = self.criterion_cls(scores_visual, target)
            l_reg_visual = (self.MSE_loss(textual_visual_embedding, video))
            l_reg_total = l_reg_visual+l_reg_audio
            l_ce_total = l_ce_visual+l_ce_audio
        elif self.xiaorong_textual_cloud == "visual+context":
            textual_visual_all_current = self.textual_enc_visual(text_embeddings_all_train[1].type(torch.float32))
            textual_context_all_current = self.textual_enc_context(text_embeddings_all_train[2].type(torch.float32))
            textual_audio_all_current, textual_visual_all_current, textual_context_all_current = self.tri_attention(
                textual_visual_all_current+textual_context_all_current, textual_visual_all_current,
                textual_context_all_current)
            textual_visual_all_current = self.textual_proj_visual(textual_visual_all_current)
            textual_context_all_current = self.textual_proj_context(textual_context_all_current)

            scores_visual = torch.matmul(video, textual_visual_all_current.t())
            scores_context = torch.matmul(video_audio, textual_context_all_current.t())
            l_ce_visual = self.criterion_cls(scores_visual, target)
            l_reg_visual = (self.MSE_loss(textual_visual_embedding, video))
            l_ce_context = self.criterion_cls(scores_context, target)
            l_reg_context =self.MSE_loss(textual_context_embedding,video_audio)
            l_reg_total = l_reg_visual + l_reg_context
            l_ce_total = l_ce_visual + l_ce_context
        elif self.xiaorong_textual_cloud == "audio+context":
            textual_audio_all_current = self.textual_enc_audio(text_embeddings_all_train[0].type(torch.float32))
            textual_context_all_current = self.textual_enc_context(text_embeddings_all_train[2].type(torch.float32))
            textual_audio_all_current, textual_visual_all_current, textual_context_all_current = self.tri_attention(
                textual_audio_all_current, textual_audio_all_current+ textual_context_all_current,
                textual_context_all_current)
            textual_audio_all_current = self.textual_proj_audio(textual_audio_all_current)
            textual_context_all_current = self.textual_proj_context(textual_context_all_current)

            scores_audio =  torch.matmul(audio, textual_audio_all_current.t())
            scores_context = torch.matmul(video_audio, textual_context_all_current.t())
            l_ce_audio = self.criterion_cls(scores_audio, target)
            l_reg_audio = (self.MSE_loss(textual_audio_embedding, audio))
            l_ce_context = self.criterion_cls(scores_context, target)
            l_reg_context = self.MSE_loss(textual_context_embedding, video_audio)
            l_reg_total = l_reg_audio + l_reg_context
            l_ce_total = l_ce_audio + l_ce_context


        elif self.xiaorong_textual_cloud == "context":
            textual_context_all_current=self.textual_enc_context(text_embeddings_all_train[2].type(torch.float32))
            _, _, textual_context_all_current = self.tri_attention(textual_context_all_current,textual_context_all_current,textual_context_all_current)
            textual_context_all_current = self.textual_proj_context(textual_context_all_current)

            scores_context=torch.matmul(video_audio, textual_context_all_current.t())
            l_ce_context = self.criterion_cls(scores_context, target)
            l_reg_context = self.MSE_loss(textual_context_embedding,video_audio)
            l_reg_total = l_reg_context
            l_ce_total = l_ce_context


        elif self.xiaorong_textual_cloud == "both":

            textual_audio_all_current=self.textual_enc_audio(text_embeddings_all_train[0])
            textual_visual_all_current=self.textual_enc_visual(text_embeddings_all_train[1].type(torch.float32))
            textual_context_all_current=self.textual_enc_context(text_embeddings_all_train[2].type(torch.float32))
            if self.xiaorong_GGA=="both":
                textual_audio_all_current, textual_visual_all_current, textual_context_all_current = self.tri_attention(
                    textual_audio_all_current, textual_visual_all_current, textual_context_all_current)

            textual_audio_all_current = self.textual_proj_audio(textual_audio_all_current)
            textual_visual_all_current = self.textual_proj_visual(textual_visual_all_current)
            textual_context_all_current = self.textual_proj_context(textual_context_all_current)

            scores_audio=torch.matmul(audio, textual_audio_all_current.t())
            scores_visual=torch.matmul(video, textual_visual_all_current.t())
            scores_context=torch.matmul(video_audio, textual_context_all_current.t())
            l_ce_audio=self.criterion_cls(scores_audio, target)
            l_ce_visual=self.criterion_cls(scores_visual, target)
            l_ce_context=self.criterion_cls(scores_context, target)
            l_ce_total=(l_ce_audio + l_ce_visual)*args.AV_coefficient + l_ce_context*args.context_coefficient

            l_reg_audio = (self.MSE_loss(textual_audio_embedding,audio))
            l_reg_visual = (self.MSE_loss(textual_visual_embedding,video))
            l_reg_context = (self.MSE_loss(textual_context_embedding,video_audio))
            l_reg_total=l_reg_audio+l_reg_visual+l_reg_context
        if args.xiaorong_loss=="ce":
            loss_total = l_ce_total
        elif args.xiaorong_loss=="reg":
            loss_total = l_reg_total
        elif args.xiaorong_loss=="both":
            loss_total = l_ce_total + l_reg_total
        loss_details = {
            "total_loss": loss_total.detach().cpu(),
            "loss_reg": l_reg_total.detach().cpu(),
            "loss_ce": l_ce_total.detach().cpu()

        }
        return loss_total, loss_details


    def optimize_params(self,
                        audio_embeddings,
                        video_embeddings,
                        targets,
                        text_embeddings,
                        textual_audio_embedding,
                        textual_visual_embedding,
                        textual_context_embedding,
                        text_embeddings_all_current,
                        args,
                        optimize=False):

        outputs = self.forward(audio_embeddings, video_embeddings, text_embeddings,textual_audio_embedding,textual_visual_embedding,textual_context_embedding)

        loss_total, loss_details = self.compute_loss(outputs, text_embeddings_all_current, targets, args,optimize)

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_total.backward()
            self.optimizer_gen.step()

        return loss_total, loss_details, self.optimizer_gen.param_groups[0]['lr']


    def get_embeddings(self, audio_embeddings, video_embeddings,textual_audio_embedding,textual_visual_embedding,textual_context_embedding):
        video_embeddings = video_embeddings.type(torch.float32)
        textual_visual_embedding=textual_visual_embedding.type(torch.float32)
        textual_context_embedding=textual_context_embedding.type(torch.float32)

        video_audio= torch.cat((video_embeddings,audio_embeddings), dim=1)
        # video_audio = torch.cat((audio_embeddings, video_embeddings), dim=1)

        video_audio=self.audio_video_enc(video_audio)
        video_embeddings = self.video_enc(video_embeddings)
        audio_embeddings = self.audio_enc(audio_embeddings)

        # model_input = torch.cat((video, audio), dim=1)

        textual_audio_embedding = self.textual_enc_audio(textual_audio_embedding)
        textual_visual_embedding = self.textual_enc_visual(textual_visual_embedding)
        textual_context_embedding = self.textual_enc_context(textual_context_embedding)
        if self.xiaorong_GGA=="both":
            textual_audio_embedding, textual_visual_embedding, textual_context_embedding = self.tri_attention(
                textual_audio_embedding, textual_visual_embedding, textual_context_embedding)

        textual_audio_embedding = self.textual_proj_audio(textual_audio_embedding)
        textual_visual_embedding = self.textual_proj_visual(textual_visual_embedding)
        textual_context_embedding = self.textual_proj_context(textual_context_embedding)

        textual_outputs = {
            "textual_audio": textual_audio_embedding,
            "textual_visual": textual_visual_embedding,
            "textual_context": textual_context_embedding,
        }

        return audio_embeddings,video_embeddings,video_audio,textual_outputs
