from torch import nn
import torch

# 构建一个余弦/正弦位置编码函数
def build_rotary_pos_emb(dim, max_token=32768, rope_base=1000000.0):
    # 构建一个从0开始，步骤为2, 直至dim的张量，并将其转换为浮点数类型  [0, 2, 4, ..., dim-2]，长度为dim//2  (dim // 2) 是为了防止dim是奇数的情况，保证最后一个位置不会越界
    base_a = torch.arange(0, dim, 2)[: (dim // 2)].float()
    
    # 归一化处理，等比例缩小到[0, 1]
    base_a = base_a / dim
    
    # rope_base专业术语叫做旋转位置编码的基数，它是一个超参数，通常设置为一个较大的值，例如1000000.0
    # 使得张量中的值，不同位置之间产生指数级差异
    base_a = rope_base ** base_a

    # 在这之前是从1到大的排列，但是我们需要从大到小的排列，因为RoPE要求的是前排位置的频率较大，后排位置的频率较小，所以我们需要将base_a进行倒数处理
    freqs = 1.0 / (rope_base ** base_a)

    # 生成一个从0到end的整数序列，长度为end，并将其放置在与freqs相同的设备上，以确保后续的计算能够在同一设备上进行
    t = torch.arange(max_token, device=freqs.device)

    # 外积计算，把张量t中的每个元素与freqs中的每个元素相乘，得到一个[end, dim//2]
    # [end, dim//2] 中的每一行表示对应位置的绝对角度，每行会有多个值，分别表示改位置向量中不同维度的旋转频率
    # 举例说明，假设我们需要计算3个token的位置编码，freqs只有两个转速： t = [0, 1, 2], freqs = [1.0, 0.1],
    # 那么外积的结果就是 [[0.0, 0.0], [1.0, 0.1], [2.0, 0.2]]，第一行表示token在绝对位置0时的不同维度的角度，第二行表示token在绝对位置1时的不同维度的角度，第三行表示token在绝对位置2时的不同维度的角度
    freqs = torch.outer(t, freqs)

    # 知道角度不是目的，我们实际上是通过正弦和余弦函数来将这些角度转换为位置编码向量的不同维度的值
    torch.cos(freqs)
    attn_factor = 1.0
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(t, cos, sin, seq_len=None):
    # 这里是一个位置编码函数，使用了旋转位置编码（Rotary Position Embedding, RoPE）的方式来为输入的张量添加位置信息
    # 旋转位置编码是一种相对于传统位置编码更为灵活和高效的位置编码方法，它通过对输入的特征向量进行旋转变换来引入位置信息
    # 具体来说，RoPE会将输入的特征向量分成两部分，一部分用于表示内容信息，另一部分用于表示位置信息，然后通过旋转变换将位置信息融入到内容信息中
    # 这样做的好处是可以在不同长度的序列上共享位置编码，同时也能够更好地捕捉长距离依赖关系

    def rotate_half(x):
        # 将指定张量切分成两部分，前一部分数据不变，后一部分全部取反
        # 并将重新将两部分数据拼接成一个张量，但是后者在前，前者在后
        """
        为什么折半截断，而不是两两组合？
        主要是为了方便显卡读取数据时的连续性，如果相邻两个特征组合，会导致CPU在计算的时候，计算完a0 a1 后，需要跳过a2 a3，才能计算a4 a5，这样就会导致CPU在读取数据的时候，无法连续读取，效率较低
        但是如果折半截断，显卡拿到的是两块连续的数据，交给GPU处理的时候，GPU在处理两部分数据的时候，可以连续读取，效率较高
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = -x[..., x.shape[-1] // 2 :]
        """
        为什么要这么做？
        张量X可以看作是： [a1, a2], 其中a1表示前半部分，a2表示后半部分

        """
        new_x = torch.cat([x2, x1], dim=-1)
        return new_x
    
    """
    为什么要这么做？
    t可以看作是[a1, a2]，其中a1表示前半部分，a2表示后半部分
    因此下面的式子可以看作是： [a1, a2] * cos + [-a2, a1] * sin = [a1*cos - a2*sin, a1*sin + a2*cos] 
    这是一个二维向量的旋转公式： [ [cos, -sin], [sin, cos]] * [a1, a2]^T 表示原向量t相对于绝对位置旋转了一个角度

    这里还有另外一个问题，为什么cos和sin要进行unsqueeze(1) 操作？
    - 目标张量 t 的 shape:   [batch, seq_len, num_attention_heads, head_dim]
    - 原始 cos/sin 的 shape: [seq_len, head_dim]
   
    执行 unsqueeze(1) 后，cos/sin 变为 [seq_len, 1, head_dim]。
    利用 PyTorch 从右向左的广播机制：
    t:   [batch, seq_len, num_heads, head_dim]
    cos: [   1 , seq_len,         1, head_dim] (最左侧的 1 是隐式补全)
   
    结论：  unsqueeze(1) 的核心目的是在 num_attention_heads 维度上占位。
            因为同一个 token 在不同 Head 下的位置编码（旋转角度）是完全一致的，
            这样操作可以把这份位置信息正确且零成本地广播给所有的 Attention Heads。
    """
    t_embed = (t * cos.unsqueeze(1)) + (rotate_half(t) * sin.unsqueeze(1))
    return t_embed


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
        freqs_cos, freqs_sin = build_rotary_pos_emb(self.head_dim, max_token=config.max_position_embeddings)

class GuluoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(6400, 512)
        self.dropout = nn.Dropout(0.0)
        
    def forward(self, inputs_ids: Optional[torch.Tensor] = None,
                past_key_values=None):
        # past_key_values用于进行K V向量的缓存，这里规定模型设置了多少层，就会有多少个缓存位置，每个位置存储一个有关KV信息的二元组
        # past_key_values是一个列表，里面中有模型层数个元素，每个元素是一个二元组，代表第i层的KV缓存信息
        # 每个二元组是两个张量构成的，分别表示这一层的K向量和V向量的缓存信息，K向量和V向量的形状都是[batch_size, sequence_length, num_heads, head_dim]
        # 其中batch_size表示批次大小，已经缓存的token数量 num_heads表示注意力头数，head_dim表示每个头的维度
        # 哪一层为None，代表这一层没有缓存信息
        # 在模型训练阶段，不需要进行KV缓存，因此默认情况下 past_key_values是None，在模型推理阶段，past_key_values会被传入具体的KV缓存信息，以便模型能够利用之前的计算结果来加速推理过程
        past_key_values = past_key_values if past_key_values is not None else [None] * len(self.layers)
        # 在模型训练阶段，初始情况下，star_pos是0
        star_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hiden_states = self.dropout(self.embed_tokens(inputs_ids))

        # 获取当前输入的tokens序列长度
        seq_len = inputs_ids.shape[1]
        end_pos = star_pos + seq_len

        # 计算位置编码
        position_embeddings = (self.freqs_cos[star_pos: end_pos, :], self.freqs_sin[star_pos: end_pos, :])
