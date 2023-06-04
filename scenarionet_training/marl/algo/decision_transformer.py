"""
Credit: https://github.com/nikhilbarhate99/min-decision-transformer/blob/master/decision_transformer/model.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# parser.add_argument('--env', type=str, default='halfcheetah')
# parser.add_argument('--dataset', type=str, default='medium')
# parser.add_argument('--rtg_scale', type=int, default=1000)
#
# parser.add_argument('--max_eval_ep_len', type=int, default=1000)
# parser.add_argument('--num_eval_ep', type=int, default=10)
#
# parser.add_argument('--dataset_dir', type=str, default='data/')
# parser.add_argument('--log_dir', type=str, default='dt_runs/')
#
# parser.add_argument('--context_len', type=int, default=20)
# parser.add_argument('--n_blocks', type=int, default=3)
# parser.add_argument('--embed_dim', type=int, default=128)  # h_dim
# parser.add_argument('--n_heads', type=int, default=1)
# parser.add_argument('--dropout_p', type=float, default=0.1)
#
# parser.add_argument('--batch_size', type=int, default=64)
# parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--wt_decay', type=float, default=1e-4)
# parser.add_argument('--warmup_steps', type=int, default=10000)
#
# parser.add_argument('--max_train_iters', type=int, default=200)
# parser.add_argument('--num_updates_per_iter', type=int, default=100)
#
# parser.add_argument('--device', type=str, default='cuda')


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        # ones = torch.ones((max_T, max_T))
        # mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        # self.register_buffer('mask', mask)

    def forward(self, data):
        x, mask = data

        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        if T > self.max_T:
            raise ValueError("Current series len {} exceeds maximum context len {}".format(T, self.max_T))

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)


        # causal mask applied to weights
        # self.mask = torch.tril(ones).view(1, 1, max_T, max_T)
        mask = mask.reshape(B, 1, 1, T).expand(-1, -1, T, -1)  # Some **columns** are set to 0.
        weights = weights.masked_fill(mask == 0, float('-inf'))

        # PZH: Do not mask anything

        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            # nn.Linear(h_dim, 2 * h_dim),  # PZH: Reduce size
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            # nn.Linear(2 * h_dim, h_dim),  # PZH: Reduce size
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, data):
        x, mask = data

        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention((x, mask))  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return (x, mask)


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, output_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        # self.act_dim = act_dim  # PZH: No one using this. Just commented out.
        self.h_dim = h_dim

        ### transformer blocks
        # input_seq_len = 3 * context_len
        # PZH: We don't need it here.
        input_seq_len = context_len + 1  # Add class token

        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        # self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # Add this
        self.cls_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        nn.init.normal_(self.cls_token, std=1e-6)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        # self.embed_action = torch.nn.Linear(act_dim, h_dim)
        # use_action_tanh = True  # True for continuous actions

        ### prediction heads
        # self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, output_dim)
        # self.predict_action = nn.Sequential(
        #     *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        # )

    def forward(self, timesteps, states, mask, actions=None, returns_to_go=None):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        # action_embeddings = self.embed_action(actions) + time_embeddings
        # returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        # h = torch.stack(
        #     (returns_embeddings, state_embeddings, action_embeddings), dim=1
        # ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        # PZH: The output h is [1, 300, 128]
        # we can make it:
        h = torch.concatenate([self.cls_token.expand(B, -1, -1), state_embeddings], dim=1)

        h = self.embed_ln(h)

        # transformer and prediction
        h, _ = self.transformer((h, mask))

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.

        # PZH: Remove 3
        # h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        h = h.reshape(B, T + 1, 1, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        # state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        # action_preds = self.predict_action(h[:, 1])  # predict action given r, s

        state_preds = self.predict_state(h[:, 0])  # Output: [bs, 1, max_T, state_dim]

        output = state_preds[:, 0]  # Output: [bs, state_dim]

        return output

class ForwardModelMaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, data):
        x, mask = data

        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        if T > self.max_T:
            raise ValueError("Current series len {} exceeds maximum context len {}".format(T, self.max_T))

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)


        # causal mask applied to weights
        # self.mask = torch.tril(ones).view(1, 1, max_T, max_T)
        # mask = mask.reshape(B, 1, 1, T).expand(-1, -1, T, -1)  # Some **columns** are set to 0.
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))

        # PZH: In forward model, we still don't have to discard future state!
        mask = mask.reshape(B, 1, 1, T).expand(-1, -1, T, -1)  # Some **columns** are set to 0.
        weights = weights.masked_fill(mask == 0, float('-inf'))

        # PZH: Do not mask anything

        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class ForwardModelBlock(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = ForwardModelMaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, data):
        x, mask = data

        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention((x, mask))  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return (x, mask)

class ForwardModelTransformer_DEPRECATED(nn.Module):
    def __init__(self, state_dim, output_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        act_dim = 2  # PZH: Hard coded here

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len

        blocks = [ForwardModelBlock(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_dynamics = torch.nn.Linear(5, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # Add this
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        # nn.init.normal_(self.cls_token, std=1e-6)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True  # True for continuous actions

        ### prediction heads
        self.predict_dynamics = torch.nn.Linear(h_dim, 5)
        self.predict_state = torch.nn.Linear(h_dim, output_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

    def forward(self, timesteps, states, actions, dynamics, mask):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        dynamics_embeddings = self.embed_dynamics(dynamics) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (dynamics_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        # PZH: The output h is [1, 300, 128]
        # we can make it:
        # h = torch.concatenate([self.cls_token.expand(B, -1, -1), state_embeddings], dim=1)

        h = self.embed_ln(h)

        # transformer and prediction
        h, _ = self.transformer((h, mask))

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.

        # PZH: Remove 3
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        # h = h.reshape(B, T + 1, 1, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        dynamics_preds = self.predict_dynamics(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict action given r, s

        # state_preds = self.predict_state(h[:, 0])  # Output: [bs, 1, max_T, state_dim]

        # output = state_preds[:, 0]  # Output: [bs, state_dim]

        return state_preds, action_preds, dynamics_preds


from vector_quantize_pytorch import VectorQuantize
class ForwardModelTransformer(nn.Module):
    def __init__(self, state_dim, output_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, codebook_size, max_timestep=4096, enable_vae=True):
        super().__init__()

        act_dim = 2  # PZH: Hard coded here

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 3 * context_len

        blocks = [ForwardModelBlock(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_dynamics = torch.nn.Linear(5, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # Add this
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, h_dim))
        # nn.init.normal_(self.cls_token, std=1e-6)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True  # True for continuous actions

        ### prediction heads
        self.predict_dynamics = torch.nn.Linear(h_dim, 5)
        self.predict_state = torch.nn.Linear(h_dim, output_dim)

        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )
        self.enable_vae = enable_vae
        self.vqvae_action = VectorQuantize(
            dim=act_dim,
            codebook_size=codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )

        self.vqvae_dynamics = VectorQuantize(
            dim=5,
            codebook_size=codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )

        self.vqvae_state = VectorQuantize(
            dim=state_dim,
            codebook_size=codebook_size,  # codebook size
            decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight=1.  # the weight on the commitment loss
        )

    def encode_vae(self, state):
        assert self.enable_vae
        quantized_state, indices_state, commit_loss_state = self.vqvae_state(state)
        return quantized_state, indices_state, commit_loss_state

        # else:
        #     return state

    # def decode_vae(self, indices):
    #     if indices.shape[-1] != 1:
    #         indices = indices.reshape(list(indices.shape) + [1, ])  # [bs, traj len, 1]
    #     codes = self.vqvae_state.get_codes_from_indices(indices)
    #     ret = self.vae_decoder(codes)
    #     return ret


    def forward(self, timesteps, states, actions, dynamics, mask):
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        if self.enable_vae:
            states, indices_state, commit_loss_state = self.vqvae_state(states)
            actions, indices_action, commit_loss_action = self.vqvae_action(actions)
            dynamics, indices_dynamics, commit_loss_dynamics = self.vqvae_dynamics(dynamics)

            loss_dict = {
                "commit_loss_state": commit_loss_state.mean(),
                "commit_loss_action": commit_loss_action.mean(),
                "commit_loss_dynamics": commit_loss_dynamics.mean(),
            }

        else:
            loss_dict = {}

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        dynamics_embeddings = self.embed_dynamics(dynamics) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (dynamics_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        # PZH: The output h is [1, 300, 128]
        # we can make it:
        # h = torch.concatenate([self.cls_token.expand(B, -1, -1), state_embeddings], dim=1)

        h = self.embed_ln(h)

        # transformer and prediction
        h, _ = self.transformer((h, mask))

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.

        # PZH: Remove 3
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        # h = h.reshape(B, T + 1, 1, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        dynamics_preds = self.predict_dynamics(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])  # predict action given r, s

        # state_preds = self.predict_state(h[:, 0])  # Output: [bs, 1, max_T, state_dim]

        # output = state_preds[:, 0]  # Output: [bs, state_dim]

        return state_preds, action_preds, dynamics_preds, loss_dict


if __name__ == '__main__':
    # Test to build the model
    model = ForwardModelTransformer(
        state_dim=29,  # 29 in our case
        output_dim=64,  # 0 in our case. Set to 2 for testing.
        n_blocks=3,
        h_dim=128,
        context_len=20,
        n_heads=1,
        drop_p=0.1,  # Dropout
        max_timestep=1000
        # Note: This is used for create the embedding of timestep. Original 4090. 200 In our case. Set to 1000 for fun.
    )


    state = torch.zeros([1, 20, 29]) + 1
    action = torch.zeros([1, 20, 2]) + 2
    dynamics = torch.zeros([1, 20, 5]) + 3
    mask = torch.ones([1, 60])
    mask[0, 30:] = 0

    timesteps = torch.from_numpy(np.arange(20)).reshape(1, -1)
    ret = model(
        timesteps, state, action, dynamics, mask
    )

    print({k: (v.shape, np.prod(v.shape)) for k, v in model.state_dict().items()})
    print(sum([np.prod(v.shape) for k, v in model.state_dict().items()]))
