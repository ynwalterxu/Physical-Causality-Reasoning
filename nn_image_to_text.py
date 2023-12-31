
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import normalize_imagenet
from transformers import ResNetConfig, ResNetModel, BertTokenizer, BertModel
import math



class NN(nn.Module):
    def __init__(self, ball, p_drop) -> None:
        super().__init__()
        self.resnet_backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
        for param in self.resnet_backbone.parameters()  :
            param.requires_grad = False
        for param in list(self.resnet_backbone.parameters())[-1:]:
            param.requires_grad = True

        self.ball = ball
        self.p_drop = p_drop

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_backbone = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert_backbone.parameters():
            param.requires_grad = False
        for param in list(self.bert_backbone.parameters())[-1:]:
            param.requires_grad = True

        # self.bert_mlp = nn.Sequential(
        #                             nn.Linear(768, 768),
        #                             nn.ReLU(),
        #                             nn.Linear(768, 768),
        #                             )

        self.resnet_mlp = nn.Sequential(
                                        nn.Linear(2048, 1024),
                                        nn.Dropout(p=self.p_drop),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.Dropout(p=self.p_drop),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.Dropout(p=self.p_drop),
                                        nn.ReLU(),
                                        nn.Linear(1024, 768)
                                        )


    def forward(self, encoded_verb_noun, image:torch.tensor, device="cuda") -> torch.tensor:
        #encoded_verb_noun = self.tokenizer(verb_noun, return_tensors='pt').to(device)
        B = image.shape[0]

        image = normalize_imagenet(image)

        x = self.bert_backbone(**encoded_verb_noun)
        x = x.pooler_output # [1, 768]
        x = x.repeat(B, 1)
        #x = self.bert_mlp(x)
        

        y = self.resnet_backbone(image)
        y = y.pooler_output # [1, 2048, 1, 1]
        y = y.reshape(B,2048)
        y = self.resnet_mlp(y)

        

        return x, y

    def loss(self, image_feats, text_feats, label):
        '''0 is similar, 1 is not similar'''
        B = image_feats.shape[0]
        zeros = torch.zeros((B, 1), device="cuda")
        distance = self.dist(text_feats, image_feats)

        # euclidean
        #loss = torch.sum( (1-label) * 0.5 * distance + label * 0.5 * torch.max(zeros, self.ball - distance) )

        # cos similarity
        loss = torch.sum( label * distance + (1-label) * torch.max(zeros, self.ball - distance) )
        return loss

    def dist(self, text_feats, image_feats):
        '''cosine similarity'''
        norm = torch.norm(text_feats, dim=1) * torch.norm(image_feats, dim=1)
        dot = torch.sum(text_feats * image_feats, dim=1)
        distance = dot / norm + 1
        distance = distance.unsqueeze(1)

        '''euclidian'''
        #distance = torch.norm(text_feats-image_feats, dim=1).unsqueeze(1)

        return distance




