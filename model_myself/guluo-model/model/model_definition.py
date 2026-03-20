from torch import nn
import torch

def apply_rotary_pos_emb(x, seq_len=None):
    # 这里是一个位置编码函数，使用了旋转位置编码（Rotary Position Embedding, RoPE）的方式来为输入的张量添加位置信息
    # 旋转位置编码是一种相对于传统位置编码更为灵活和高效的位置编码方法，它通过对输入的特征向量进行旋转变换来引入位置信息
    # 具体来说，RoPE会将输入的特征向量分成两部分，一部分用于表示内容信息，另一部分用于表示位置信息，然后通过旋转变换将位置信息融入到内容信息中
    # 这样做的好处是可以在不同长度的序列上共享位置编码，同时也能够更好地捕捉长距离依赖关系
    return x


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义自注意力机制的多头注意力层
        self.num_key_value_heads = 8
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # Dropout层：它会在训练过程中随机将一些神经元的输出设置为零，以防止过拟合
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self, x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor]):
        # x是一个张量，进来的形状是[batch_size, sequence_length, hidden_size]
        # 其中 batch_size表示批次，以处理句子为例，batch_size标识一次性总共处理的句子数量, 不同的句子互相独立
        # squence_length表示在处理每个句子的时候，一次性处理的token数量
        # hiden_size表示每个token的特征向量的维度是多少，可以简单的理解为用一个多大的向量来表示一个token
        bsz, seq_len, _ = x.shape
        # 计算Q K V
        x_q = self.q_proj(x)
        x_k = self.k_proj(x)
        x_v = self.v_proj(x)

        # 以Q张量为例，他的形状是[batch_size, sequence_length, hidden_size]
        # 在当前注意力机制中，我们采用多头注意力机制，这里并非使用经典的多头注意力机制，而是使用了Grouped Query Attention (GQA)的机制，GQA的机制是将Q的头数分成多个组，每个组对应一个KV头
        # 因此我们需要根据注意力头数，将hidden_size维度分成多个头，每个头的维度是head_dim
        # 因此Q张量需要被展开为四维张量，形状为[batch_size, sequence_length, num_attention_heads, head_dim]
        # view()函数的作用就是将已知张量进行维度重塑和逻辑切分
        x_q = x_q.view(bsz, seq_len, self.num_attention_heads, self.head_dim)

        # 这是因为在这里我们采用了Grouped Query Attention (GQA)的机制，GQA的机制是将Q的头数分成多个组，每个组对应一个KV头
        # 因此比例变成了：N:1:1的关系
        x_k = x_k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        x_v = x_v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # RoPE位置编码
        # 在实际的实现中，必须为Q和K张量应用旋转位置编码（RoPE），以便为输入的特征向量添加位置信息
        # 如果没有位置信息的话，对于语句 “我吃苹果” 和 “苹果吃我”，模型就无法区分他们的差异，将其当作同一句话来处理了
        x_q = apply_rotary_pos_emb(x_q, seq_len=seq_len)
        x_k = apply_rotary_pos_emb(x_k, seq_len=seq_len)







# 我想定义一个Decoder only架构的Block层
class GuluoBlock(nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

class GuluoModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass