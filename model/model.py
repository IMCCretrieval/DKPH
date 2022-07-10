import torch
import torch.nn as nn
from torch.autograd import Variable, Function

import sys 
sys.path.append("/home/lipd/DKPH/utils/") #notice that the path need be changed when the filename is renamed.
from args import nbits,hidden_size
from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward

class DKPH(nn.Module):
    """
    Dual-stream Knowledge-Preserving hashing Model
    """
    def __init__(self, frame_size):

        super(DKPH,self).__init__()
        self.bert = BERT(frame_size, hidden=hidden_size, n_layers=1, attn_heads=1, dropout=0.1)
        self.hid_b = nn.Linear(hidden_size*25, nbits)
        self.temporal = nn.Linear(hidden_size, nbits)
        self.hash  = self.binarization
        self.decoder = nn.Linear(nbits, frame_size)

    def binarization(self,x):
        y = (x+1.)/2.
        y[y>1] = 1
        y[y<0] = 0    
        y = 2.*BNet.apply(y)-1.   
        return y
    
    def forward(self, x):
        embeddings = self.bert(x)  
        tt = self.temporal(embeddings) # latent features

        hid_b = self.hid_b(embeddings.reshape([embeddings.shape[0],embeddings.shape[1]*embeddings.shape[2]]))  
        bb = self.hash(hid_b) # binary codes 
        
        recon_frames = self.decoder(torch.unsqueeze(bb,dim=1).repeat(1,25,1)+tt)
        
        return bb, recon_frames, embeddings 


class BNet(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = 1-(ctx.input==0)  # mask 0 
        mask = Variable(mask).cuda().float()  
        grad_output = grad_output*mask
        return grad_output, None, None

class BERT(nn.Module):
    """
    BERT model  
    """

    def __init__(self, frame_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(BERT,self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(frame_size=frame_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)

        return x

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
