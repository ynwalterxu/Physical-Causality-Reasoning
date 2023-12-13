
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetConfig, ResNetModel, BertTokenizer, BertModel
import math



class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet_backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
        for param in self.resnet_backbone.parameters()  :
            param.requires_grad = False
        # for param in list(self.resnet_backbone.parameters())[-1:]:
        #     param.requires_grad = True

        self.ball = 1.0

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_backbone = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert_backbone.parameters():
            param.requires_grad = False
        # for param in list(self.bert_backbone.parameters())[-1:]:
        #     param.requires_grad = True

        self.bert_mlp = nn.Sequential(
                                        nn.Linear(768, 1024)
                                    )

        self.resnet_mlp = nn.Sequential(
                                        nn.Linear(2048, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 768),
                                        )

        self.gpt = GPT(n_embd=2048)




    def forward(self, encoded_verb_noun, image:torch.tensor, device="cuda") -> torch.tensor:
        #encoded_verb_noun = self.tokenizer(verb_noun, return_tensors='pt').to(device)
        B = image.shape[0]

        x = self.bert_backbone(**encoded_verb_noun)
        x = x.pooler_output # [1, 768]
        x = x.repeat(B, 1)
        #x = self.bert_mlp(x)
        

        y = self.resnet_backbone(image)
        y = y.pooler_output # [1, 2048, 1, 1]
        y = y.reshape(B,2048)
        y = self.resnet_mlp(y)

        #z = self.gpt(y, x)

        #image_tensor = z[:,:1024]
        #text_tensor = z[:,1024:]

        

        return x, y#image_tensor, text_tensor

    def loss(self, image_feats, text_feats, label):
        '''0 is similar, 1 is not similar'''
        B = image_feats.shape[0]
        zeros = torch.zeros((B, 1), device="cuda")
        distance = self.dist(text_feats, image_feats)

        # euclidean
        #loss = torch.sum( (1-label) * 0.5 * distance + label * 0.5 * torch.max(zeros, self.ball - distance) )

        # cos similarity
        loss = torch.sum( label * 0.5 * distance + (1-label) * 0.5 * torch.max(zeros, self.ball - distance) )

        

        return loss

    def dist(self, text_feats, image_feats):
        '''cosine similarity'''
        norm = torch.norm(text_feats, dim=1) * torch.norm(image_feats, dim=1)
        dot = torch.sum(text_feats * image_feats, dim=1)
        distance = dot / norm + 1
        distance = distance.unsqueeze(1)

        '''euclidian'''
        # distance = torch.norm(text_feats-image_feats, dim=1).unsqueeze(1)

        return distance

class GPT(nn.Module):
  """  the full GPT language backbone, with a context size of block_size """

  def __init__(self, n_embd, n_head=4, n_layer=4, block_exp=2, attn_pdrop=0.1, resid_pdrop=0.1, embd_pdrop=0.1):
    super().__init__()
    self.n_embd = n_embd

    # positional embedding parameter (learnable), image + lidar
    self.pos_emb = nn.Parameter(
        torch.zeros(1, self.n_embd)
        )

    self.drop = nn.Dropout(embd_pdrop)

    # transformer
    self.blocks = nn.Sequential(*[
        Block(n_embd, n_head, block_exp, attn_pdrop, resid_pdrop)
        for layer in range(n_layer)
    ])

    # decoder head
    self.ln_f = nn.LayerNorm(n_embd)

    #self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean, std=self.config.gpt_linear_layer_init_std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

  def forward(self, image_tensor, text_tensor):

    B = text_tensor.shape[0]

    token_embeddings = torch.cat((image_tensor, text_tensor), dim=1)

    x = self.drop(self.pos_emb + token_embeddings)
    x = self.blocks(x)  # (B, an * T, C)
    x = self.ln_f(x)  # (B, an * T, C)

    # image_tensor_out = x[:, :self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors, :].view(
    #     bz * self.seq_len, img_h, img_w, -1).permute(0, 3, 1, 2).contiguous()

    # lidar_tensor_out = x[:, self.seq_len * self.config.img_vert_anchors * self.config.img_horz_anchors:, :].view(
    #       bz, lidar_h, lidar_w, -1).permute(0, 3, 1, 2).contiguous()

    return x


class SelfAttention(nn.Module):
  """
    A vanilla multi-head masked self-attention layer with a projection at the
    end.
    """

  def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
    super().__init__()
    assert n_embd % n_head == 0
    # key, query, value projections for all heads
    self.key = nn.Linear(n_embd, n_embd)
    self.query = nn.Linear(n_embd, n_embd)
    self.value = nn.Linear(n_embd, n_embd)
    # regularization
    self.attn_drop = nn.Dropout(attn_pdrop)
    self.resid_drop = nn.Dropout(resid_pdrop)
    # output projection
    self.proj = nn.Linear(n_embd, n_embd)
    self.n_head = n_head

  def forward(self, x):
    B, D = x.shape

    k = self.key(x).reshape(B, self.n_head, D//self.n_head)
    q = self.query(x).reshape(B, self.n_head, D//self.n_head)
    v = self.value(x).reshape(B, self.n_head, D//self.n_head)

    att = q * k / math.sqrt(k.shape[-1])
    att = F.softmax(att, dim=-1)
    att = self.attn_drop(att)
    y = att * v 
    y = y.reshape(B, D)

    # output projection
    y = self.resid_drop(self.proj(y))
    return y

class Block(nn.Module):
  """ an unassuming Transformer block """

  def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
    super().__init__()
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
    self.mlp = nn.Sequential(
        nn.Linear(n_embd, block_exp * n_embd),
        nn.ReLU(True),  # changed from GELU
        nn.Linear(block_exp * n_embd, n_embd),
        nn.Dropout(resid_pdrop),
    )

  def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))

    return x





