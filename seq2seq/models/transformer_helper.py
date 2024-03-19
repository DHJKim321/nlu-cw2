import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-6-DESCRIBE-D-START___
        1.  What is the purpose of encoder_padding_mask?
            The encoder padding mask ensures that the mask is set to -inf for any padding tokens.
            Therefore, the final attention weights for each padding will be -inf, which in turn means that
            the encoder does not attend to these tokens when calculating the encoded contexts.
        '''
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)
        '''
        ___QUESTION-6-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                key=state,
                                value=state,
                                key_padding_mask=self_attn_padding_mask,
                                need_weights=False,
                                attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-6-DESCRIBE-E-START___
        1.  How does encoder attention differ from self attention?
            In self attention, the sizes of q, k, v are the same as they come from the same place
            (output of the previous encoder layer).
            On the other hand, in encoder-decoder attention, 
            q comes from the previous decoder layer, while k and v comes from the output of the encoder.
        2.  What is the difference between key_padding_mask and attn_mask?
            attn_mask deals with limiting the model's knowledge of future tokens. I.e., limiting leftward information flow.
            Whereas, key_padding_mask deals with masking the attention given to the padding tokens.
        3.  If you understand this difference, then why don't we need to give attn_mask here?
            Since we are dealing with contexts passed from the encoder, 
            we do not want to limit leftward information flow. Therefore, we want each target token to interact with all source tokens.
        '''
        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        '''
        ___QUESTION-6-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                embed_dim,
                num_attn_heads,
                kdim=None,
                vdim=None,
                dropout=0.,
                self_attention=False,
                encoder_decoder_attention=False):
        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        src_time_steps = key.size(0)

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, src_time_steps]
        attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, src_time_steps))

        # First need to perform linear projection of Q, K, and V
        projected_q = self.q_proj(query)
        print(projected_q.size())
        # query.size = [tgt_time_steps, batch_size, self.embed_dim]
        # If self-attention:
        #   self.q_proj = [self.embed_dim, self.embed_dim]
        #   projected_q.size = [tgt_time_steps, batch_size, self.embed_dim]
        # Else if cross-attention: (self.k_embed_size == self.v_embed_size == self.embed_dim in most cases)
        #   self.q_proj = [self.k_embed_size, self.embed_dim]
        #   projected_q.size = [tgt_time_steps, batch_size, self.embed_dim]

        # Similarly:
        projected_k = self.k_proj(key)
        print(projected_k.size())
        # projected_k.size = [src_time_steps, batch_size, self.embed_dim]
        projected_v = self.v_proj(value)
        # projected_v.size = [src_time_steps, batch_size, self.embed_dim]

        # Transpose projected q, k, v into [batch_size, time_steps, self.embed_dim]
        projected_q = projected_q.transpose(0, 1)
        # projected_q.size = [batch_size, tgt_time_steps, self.embed_dim]
        projected_k = projected_k.transpose(0, 1)
        # projected_k.size = [batch_size, src_time_steps, self.embed_dim]
        projected_v = projected_v.transpose(0, 1)
        # projected_v.size = [batch_size, src_time_steps, self.embed_dim]

        # Split projected q, k, v into self.num_heads heads

        reshaped_projected_q = projected_q.reshape(batch_size, tgt_time_steps, self.num_heads, self.head_embed_size)
        # reshaped_projected_q.size = [batch_size, tgt_time_steps, self.num_heads, self.head_embed_size]
        reshaped_projected_k = projected_k.reshape(batch_size, src_time_steps, self.num_heads, self.head_embed_size)
        # reshaped_projected_k.size = [batch_size, src_time_steps, self.num_heads, self.head_embed_size]
        reshaped_projected_v = projected_v.reshape(batch_size, src_time_steps, self.num_heads, self.head_embed_size)
        # reshaped_projected_v.size = [batch_size, src_time_steps, self.num_heads, self.head_embed_size]

        head_projected_q = reshaped_projected_q.transpose(1, 2)
        # head_projected_q.size = [batch_size, self.num_heads, tgt_time_steps, self.head_embed_size]
        head_projected_k = reshaped_projected_k.transpose(1, 2)
        # head_projected_k.size = [batch_size, self.num_heads, src_time_steps, self.head_embed_size]
        head_projected_v = reshaped_projected_v.transpose(1, 2)
        # head_projected_v.size = [batch_size, self.num_heads, src_time_steps, self.head_embed_size]
    
        # Fold each head into the batch dimension
        # I.e., Treat each x in batch*self.num_heads as its own batch.
        batched_q = head_projected_q.reshape(batch_size * self.num_heads, tgt_time_steps, self.head_embed_size)
        # batched_q.size = [batch_size * self.num_heads, tgt_time_steps, self.head_embed_size]
        batched_k = head_projected_k.reshape(batch_size * self.num_heads, src_time_steps, self.head_embed_size)
        # batched_k.size = [batch_size * self.num_heads, src_time_steps, self.head_embed_size]
        batched_v = head_projected_v.reshape(batch_size * self.num_heads, src_time_steps, self.head_embed_size)
        # batched_v.size = [batch_size * self.num_heads, src_time_steps, self.head_embed_size]

        # Calculate the attention weights
        transposed_batched_k = batched_k.transpose(1, 2)
        # transposed_batched_k.size = [batch_size * self.num_heads, self.head_embed_size, src_time_steps]

        raw_scores = torch.bmm(batched_q, transposed_batched_k)
        # raw_scores.size = [batch_size * self.num_heads, tgt_time_steps, src_time_steps]

        scaled_scores = raw_scores / self.head_scaling
        # scaled_scores.size = [batch_size * self.num_heads, tgt_time_steps, src_time_steps]

        # If masks are provided:
        # Mask value must be -inf for all values in the input of the softmax according to Vaswani et al. 2017.
        # This ensures that the subsequent softmax operation will result in 0 for all masked values.

        # attn_mask.size = [tgt_time_steps, tgt_time_steps], attn_mask.type = float (-inf)
        # Decoder-only as we want to prevent leftward information flow.
        # tgt_time_steps is a constant value (i.e., maximum input length for decoder transformer)

        if attn_mask is not None:
            attention_mask = attn_mask.unsqueeze(dim=0)
            # attention_mask.size = [1, tgt_time_steps, tgt_time_steps]
            scaled_scores += attention_mask
            # scaled_scores.size = [batch_size * self.num_heads, tgt_time_steps, src_time_steps]
            # If scaled_scores + attention_mask, they should have the same size. batch_size * self.num_heads =\ 1 so that should modifiy it. Currently, attn_maks = None for our test. 
    

        # key_padding_mask.size = [batch_size, src_time_steps], key_padding_mask.type = Boolean
        # True if padding, False if not padding.
        if key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(dim=1)
            # padding_mask.size = [batch_size, 1, src_time_steps]
            # Repeat padding mask for each head
            # Can leave other dimensions to be broadcasted
            padding_mask = padding_mask.repeat(self.num_heads, 1, 1)
            # padding_mask.size = [batch_size * self.num_heads, 1, src_time_steps]
            scaled_scores = scaled_scores.masked_fill(padding_mask, float('-inf'))
            # transposed_softmax_scores.size = [batch_size * self.num_heads, tgt_time_steps, src_time_steps]

        batched_softmax_scores = F.softmax(scaled_scores, dim=2)
        # batched_softmax_scores.size = [batch_size * self.num_heads, tgt_time_steps, src_time_steps]

        # Perform dropout after calculating attention weights
        batched_softmax_scores = F.dropout(batched_softmax_scores, p=self.attention_dropout, training=self.training)
        # batched_softmax_scores.size = [batch_size * self.num_heads, tgt_time_steps, src_time_steps]

        unfolded_weights = batched_softmax_scores.reshape(batch_size, self.num_heads, tgt_time_steps, src_time_steps)
        # unfolded_weights.size = [batch_size, self.num_heads, tgt_time_steps, src_time_steps]

        # Store to attn_weights if needed, ensure that dimensions are correct
        attn_weights += unfolded_weights.transpose(0, 1) if need_weights else None
        # attn_weights.size = [self.num_heads, batch_size, tgt_time_steps, src_time_steps]

        # Calculate weighted attention values
        batched_attn = torch.bmm(batched_softmax_scores, batched_v)
        # batched_attn.size = [batch_size * self.num_heads, tgt_time_steps, self.head_embed_size]

        temp_attn_weights = batched_attn.reshape(batch_size, self.num_heads, tgt_time_steps, self.head_embed_size)
        # temp_attn.size = [batch_size, self.num_heads, tgt_time_steps, self.head_embed_size]

        transposed_attn = temp_attn_weights.transpose(1, 2)
        # transposed_attn.size = [batch_size, tgt_time_steps, self.num_heads, self.head_embed_size]

        # Merge the heads back
        concat_attn = transposed_attn.reshape(batch_size, tgt_time_steps, self.embed_dim)
        # concat_attn.size = [batch_size, tgt_time_steps, self.embed_dim]

        # Project the concatenated attention values (W^{o})
        final_projected_attn = self.out_proj(concat_attn)
        # final_projected_attn.size = [batch_size, tgt_time_steps, self.embed_dim]

        transposed_projected_attn = final_projected_attn.transpose(0, 1)
        # transposed_projected_attn.size = [tgt_time_steps, batch_size, self.embed_dim]

        # Add the projected attention values to the attn tensor and ensure that dimensions are correct
        attn += transposed_projected_attn
        # attn.size = [tgt_time_steps, batch_size, self.embed_dim]

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        '''

        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m
