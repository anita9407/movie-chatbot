import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.autograd import Variable

from . import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_STATE_SIZE = 512
EMBEDDING_DIM = 50
Dropout=0.5

"""
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention,self).__init__()
        self.enc_hid_dim=enc_hid_dim
        self.dec_hid_dim=dec_hid_dim
        self.attn=nn.Linear((enc_hid_dim*2)+dec_hid_dim,dec_hid_dim)
        self.v=nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        batch_size=encoder_outputs.shape[1]
        src_len=encoder_outputs.shape[0]
        #重复操作，让隐藏状态的第二个维度和encoder相同
        hidden=hidden.unsqueeze(1).repeat(1,src_len,1)
        #该函数按指定的向量来重新排列一个数组，在这里是调整encoder输出的维度顺序，在后面能够进行比较
        encoder_outputs=encoder_outputs.permute(1,0,2)
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        #开始计算hidden和encoder_outputs之间的匹配值
        energy=torch.tanh(self.attn(torch.cat((hidden,encoder_outputs),dim=2)))
        #energy = [batch size, src sent len, dec hid dim]
        #调整energy的排序
        energy=energy.permute(0,2,1)
        #energy = [batch size, dec hid dim, src sent len]
        
        #v = [dec hid dim]
        v=self.v.repeat(batch_size,1).unsqueeze(1)
        #v = [batch_size, 1, dec hid dim] 注意这个bmm的作用，对存储在两个批batch1和batch2内的矩阵进行批矩阵乘操
        attention=torch.bmm(v,energy).squeeze(1)
        #attention=[batch_size, src_len]
        return F.softmax(attention, dim=1) 
"""  

class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True,bidirectional=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True,bidirectional=True)
        self.output = nn.Linear(hid_size*2, dict_size)
        
    def forward(self, x):       #x=[src sent len, batch size]
        embed = self.emb(x)     #embed = [src sent len, batch size, emb dim]       
        embed = torch.transpose(embed, dim0=1, dim1=0) #embed = [batch size，src sent len，emb dim]
        out,(h_n,c_n) = self.encoder(embed)
        final_out = self.output(torch.cat([c_n[i,:,:]for i in range(c_n.shape[0])], dim=1))
        # 前向传播 LSTM
        #fw_out, hidden = self.encoder(x, (h0, c0))  # LSTM输出大小为 (batch_size, seq_length, hidden_size*2)
        # 解码最后一个时刻的隐状态
        #fw_out = self.output(out[:, -1, :])
        #final_out = self.output(fw_out[:, -1, :])
        
        #hidden = torch.cat(self.output(hidden[-2,:,:]),self.output(hidden[-1,:,:]))
        return final_out
          
    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        # return encoded[:, index:index+1]
        # For LSTM
        return encoded[0][:, index:index+1].contiguous(), \
               encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self, hid, input_seq):
        # Method assumes batch of size=1
        out, _ = self.decoder(input_seq, hid)
        out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x):
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        out = self.output(out)
        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again. Act greedily
        """
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = torch.max(out_logits, dim=1)[1]
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v)

            res_logits.append(out_logits)
            res_tokens.append(out_token)
            if stop_at_token is not None and out_token == stop_at_token:
                break
        return torch.cat(res_logits), res_tokens

    def decode_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None):
        """
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        """
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]
            action = int(np.random.choice(out_probs.shape[0], p=out_probs))
            action_v = torch.LongTensor([action]).to(begin_emb.device)
            cur_emb = self.emb(action_v)

            res_logits.append(out_logits)
            res_actions.append(action)
            if stop_at_token is not None and action == stop_at_token:
                break
        return torch.cat(res_logits), res_actions


def pack_batch_no_out(batch, embeddings, device="cpu"):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    lens = list(map(len, input_idx))
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    input_v = torch.tensor(input_mat).to(device)
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    # lookup embeddings
    r = embeddings(input_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
    return emb_input_seq, input_idx, output_idx


def pack_input(input_data, embeddings, device="cpu"):
    input_v = torch.LongTensor([input_data]).to(device)
    r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)


def pack_batch(batch, embeddings, device="cpu"):
    emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)

    # prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        output_seq_list.append(pack_input(out[:-1], embeddings, device))
    return emb_input_seq, output_seq_list, input_idx, output_idx


def seq_bleu(model_out, ref_seq):
    model_seq = torch.max(model_out.data, dim=1)[1]
    model_seq = model_seq.cpu().numpy()
    return utils.calc_bleu(model_seq, ref_seq)

"""
class PhraseModel(nn.Module):
    def __init__(self, emb_size, dict_size, hid_size):
        super(PhraseModel, self).__init__()

        self.emb = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
        self.encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True,bidirectional=True)
        self.decoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size,
                               num_layers=1, batch_first=True,bidirectional=True)
        self.output = nn.Sequential(
            nn.Linear(hid_size*2, dict_size)
        )
        
    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x,self.hid_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2,x,self.hid_size).to(self.device)
        
        out,_ = self.encoder(x,(h0,c0))
        out = self.output(out[:,-1:])
        return out
    
    def encode(self, x):
        _, hid = self.encoder(x)
        return hid

    def get_encoded_item(self, encoded, index):
        # For RNN
        # return encoded[:, index:index+1]
        # For LSTM
        return encoded[0][:, index:index+1].contiguous(), \
               encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self, hid, input_seq):
        # Method assumes batch of size=1
        out, _ = self.decoder(input_seq, hid)
        out = self.output(out.data)
        return out

    def decode_one(self, hid, input_x):
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        out = self.output(out)
        return out.squeeze(dim=0), new_hid

    def decode_chain_argmax(self, hid, begin_emb, seq_len, stop_at_token=None):
        
        Decode sequence by feeding predicted token to the net again. Act greedily
        
        res_logits = []
        res_tokens = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_token_v = torch.max(out_logits, dim=1)[1]
            out_token = out_token_v.data.cpu().numpy()[0]

            cur_emb = self.emb(out_token_v)

            res_logits.append(out_logits)
            res_tokens.append(out_token)
            if stop_at_token is not None and out_token == stop_at_token:
                break
        return torch.cat(res_logits), res_tokens

    def decode_chain_sampling(self, hid, begin_emb, seq_len, stop_at_token=None):
        
        Decode sequence by feeding predicted token to the net again.
        Act according to probabilities
        
        res_logits = []
        res_actions = []
        cur_emb = begin_emb

        for _ in range(seq_len):
            out_logits, hid = self.decode_one(hid, cur_emb)
            out_probs_v = F.softmax(out_logits, dim=1)
            out_probs = out_probs_v.data.cpu().numpy()[0]
            action = int(np.random.choice(out_probs.shape[0], p=out_probs))
            action_v = torch.LongTensor([action]).to(begin_emb.device)
            cur_emb = self.emb(action_v)

            res_logits.append(out_logits)
            res_actions.append(action)
            if stop_at_token is not None and action == stop_at_token:
                break
        return torch.cat(res_logits), res_actions


def pack_batch_no_out(batch, embeddings, device="cpu"):
    assert isinstance(batch, list)
    # Sort descending (CuDNN requirements)
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    input_idx, output_idx = zip(*batch)
    # create padded matrix of inputs
    lens = list(map(len, input_idx))
    input_mat = np.zeros((len(batch), lens[0]), dtype=np.int64)
    for idx, x in enumerate(input_idx):
        input_mat[idx, :len(x)] = x
    input_v = torch.tensor(input_mat).to(device)
    input_seq = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    # lookup embeddings
    r = embeddings(input_seq.data)
    emb_input_seq = rnn_utils.PackedSequence(r, input_seq.batch_sizes)
    return emb_input_seq, input_idx, output_idx


def pack_input(input_data, embeddings, device="cpu"):
    input_v = torch.LongTensor([input_data]).to(device)
    r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r, [len(input_data)], batch_first=True)


def pack_batch(batch, embeddings, device="cpu"):
    emb_input_seq, input_idx, output_idx = pack_batch_no_out(batch, embeddings, device)

    # prepare output sequences, with end token stripped
    output_seq_list = []
    for out in output_idx:
        output_seq_list.append(pack_input(out[:-1], embeddings, device))
    return emb_input_seq, output_seq_list, input_idx, output_idx


def seq_bleu(model_out, ref_seq):
    model_seq = torch.max(model_out.data, dim=1)[1]
    model_seq = model_seq.cpu().numpy()
    return utils.calc_bleu(model_seq, ref_seq)

"""
