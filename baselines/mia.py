def get_output_dims(modes):
    dims_out = {}
    for mode in modes:
        if mode in ["rgb", "curvature", "normal", "reshading", "sketch"]:
            dims_out[mode] = 3
        elif mode in ["depth"]:
            dims_out[mode] = 1
        elif mode in ["semseg"]:
            dims_out[mode] = 19
        else:
            dims_out[mode] = 1

    return dims_out


def get_input_dims(modes):
    dims_out = {}
    for mode in modes:
        if mode in ["rgb", "curvature", "normal", "reshading", "depth", "sketch", "semseg"]:
            dims_out[mode] = 2
        elif mode in ["temperature", "pressure", "precipitation", "humidity"]:
            dims_out[mode] = 2
        else:
            dims_out[mode] = 1

    return dims_out


def get_out_bias(mode):
    if mode in ["rgb", "curvature", "normal", "reshading", "depth", "sketch"]:
        return 0.5
    elif mode in ["temperature", "pressure", "precipitation", "humidity"]:
        return 0.5
    else:
        return 0.0


def get_input_range(mode):
    if mode in ["sine", "tanh", "gaussian", "relu", "wav"]:
        return 1.0
    else:
        return 1.0


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from math import sqrt


# Helpers
# ------------------------------------------------------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# Transformers, Perceivers
# ------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, topk=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        sim = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask.bool(), max_neg_value)

        if topk is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            B, nH, nQ, nK = sim.shape
            topk = min(topk, nK)
            _, topk_idx = torch.topk(sim, topk, sorted=False, dim=-1)
            topk_mask = torch.ones_like(sim, dtype=torch.bool)
            topk_mask[
                torch.arange(B, device=sim.device).view(B, 1, 1, 1),
                torch.arange(nH, device=sim.device).view(1, nH, 1, 1),
                torch.arange(nQ, device=sim.device).view(1, 1, nQ, 1),
                topk_idx,
            ] = False
            sim.masked_fill_(topk_mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, mask=None, topk=None):
        for attn, ffn in self.layers:
            x = attn(x, mask=mask, topk=topk)
            x = ffn(x)
        return x


class Perceiver(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        num_querys,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.cross_attn_layer = nn.ModuleList(
            [
                PreNorm(
                    dim,
                    Attention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    context_dim=context_dim,
                ),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]
        )

        self.self_attn_layers = nn.ModuleList()
        for _ in range(depth):
            self.self_attn_layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

        self.num_querys = num_querys
        self.querys = nn.Parameter(torch.randn(1, num_querys, dim) * 0.2)

    def forward(self, context, mask=None, topk=None):
        attn, ffn = self.cross_attn_layer

        x = repeat(self.querys, "1 n d -> b n d", b=context.shape[0])
        x = attn(x, context=context, mask=mask, topk=topk) + x
        x = ffn(x) + x

        for attn, ffn in self.self_attn_layers:
            x = attn(x)
            x = ffn(x)
        return x


# Fourier Features
# ------------------------------------------------------------------------
class LearnableFourierFeatures(nn.Module):
    def __init__(self, xdim, fdim):
        super().__init__()
        self.fdim = fdim
        self.mat = nn.Linear(xdim, fdim // 2, bias=False)

    def forward(self, x):
        x = self.mat(x)
        x = torch.cat([torch.cos(x), torch.sin(x)], -1) * 1 / sqrt(self.fdim)
        return x


class FixedFourierFeatures(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        w = torch.exp(torch.linspace(0, 8, fdim // 2))
        self.register_buffer("w", w)

    def forward(self, x):
        x = einsum(x, self.w, "... d, fdim -> ... d fdim")
        x = torch.pi * rearrange(x, "... d fdim -> ... (d fdim)")
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class RandomFourierFeatures(nn.Module):
    def __init__(self, fdim, in_dim, sigma):
        super().__init__()
        B = torch.randn(fdim // 2, in_dim) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        """
        x : (bsz, n, in_dim)
        self.B : (fdim // 2, in_dim)
        return : (bsz, n, fdim)
        """
        x = 2 * torch.pi * x @ self.B.T
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


# Activations
# ------------------------------------------------------------------------
class Sine(nn.Module):
    def __init__(self, w0=1.0, w0_learnable=False):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0), requires_grad=w0_learnable)

    def forward(self, x):
        return torch.sin(self.w0 * x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class LatentReshape1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        b, mss, d = x.shape
        x = rearrange(x, "b l d -> b d l")
        return x


class ModulateReshape1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        b, mss, d = x.shape
        x = rearrange(x, "b d l -> b l d")
        return x


class LatentReshape2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        b, mss, d = x.shape
        if sqrt(mss) == int(sqrt(mss)):
            h = w = int(sqrt(mss))
        else:
            # this is the case when H:W = 1:2
            h = int(sqrt(mss / 2))
            w = int(h * 2)
        x = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
        return x


class ModulateReshape2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        depth=2,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if depth == 1:
            layers = [nn.Linear(in_features, out_features)]
        elif depth > 1:
            layers = [nn.Linear(in_features, hidden_features)]
            for _ in range(depth - 2):
                layers += [act_layer(), nn.Linear(hidden_features, hidden_features)]
            layers += [act_layer(), nn.Linear(hidden_features, out_features)]
        else:
            layers = [nn.Identity()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


import torch

def naive_grid_sample_2d(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)


    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def naive_grid_sample_1d(input, grid):
    ix = grid[..., 0]
    N, C, IL = input.shape
    _, L, _ = grid.shape

    ix = ((ix + 1) / 2) * (IL-1);
    with torch.no_grad():
        ix_w = torch.floor(ix);
        ix_e = ix_w + 1;

    nw = (ix_e - ix).view(N, 1, L)
    ne = (ix   - ix_w).view(N, 1, L)

    with torch.no_grad():
        torch.clamp(ix_w, 0, IL-1, out=ix_w)
        torch.clamp(ix_e, 0, IL-1, out=ix_e)

    w_val = torch.gather(input, 2, ix_w.long().view(N, 1, L).repeat(1, C, 1))
    e_val = torch.gather(input, 2, ix_e.long().view(N, 1, L).repeat(1, C, 1))

    out_val = w_val * nw + e_val * ne
    return out_val



from einops import rearrange

def grid_sample(input, coords):
    if coords.shape[-1] == 1:
        # 1D case (synthetic, audio)
        # input shape       (B, D, L)
        # coords shape      (B, N, 1)
        # output shape      (B, D, N)
        output = naive_grid_sample_1d(input, coords)
        output = rearrange(output, 'b d n -> b n d')
        return output

    elif coords.shape[-1] == 2:
        # 2D case (image)
        # input shape       (B, D, H, W)
        # coords shape      (B, N, 2)
        # coords -> grid    (B, 1, N, 2)
        grid = coords.unsqueeze(1)[..., [1, 0]]
        output = naive_grid_sample_2d(input, grid)
        output = rearrange(output, 'b d 1 n -> b n 1 d').squeeze(2)
        return output
    else:
        raise NotImplementedError('Only 1D, 2D cases are implemented.')


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, einsum
from math import sqrt



class SirenLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        w0_learnable=False,
        use_bias=True,
        c=6.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # initialize layers following SIREN paper
        w_std = sqrt(c / dim_in) / w0
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0, w0_learnable)

    def forward(self, x, scale=1.0, shift=0.0):
        return self.activation(scale * self.linear(x) + shift)

    def forward_lowrank_gate(self, x, u_mat, v_mat):
        """lowrank gate modulation

        Args:
            x (torch.Tensor): input [batch_size, ..., dim_in]
            u_mat (torch.Tensor): U matrix [batch_size, rank, dim_out]
            v_mat (torch.Tensor): V matrix [batch_size, rank, dim_in]

        Returns:
            torch.Tensor: activation((sigmoid(UV) * W)x + b) [batch_size, ..., dim_out]
        """
        rank = u_mat.shape[1]
        G = einsum(u_mat, v_mat, "b r do, b r di -> b do di") / sqrt(rank)
        G = torch.sigmoid(G)

        modulated_weight = G * self.linear.weight.unsqueeze(0)
        bias = 0.0 if self.linear.bias is None else self.linear.bias
        out = einsum(modulated_weight, x, "b o i, b ... i -> b ... o")
        out = out + bias
        return self.activation(out)


class SirenLayerFirst(SirenLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        w0_learnable=False,
        use_bias=True,
    ):
        super().__init__(
            dim_in,
            dim_out,
            w0,
            w0_learnable,
            use_bias,
        )

        # initialize layers following SIREN paper
        w_std = 1 / dim_in
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)


class INRLayerLast(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.out_bias = out_bias

    def forward(self, x):
        return self.linear(x) + self.out_bias


class INRLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = nn.ReLU()

    def forward(self, x, scale=1.0, shift=0.0):
        return self.activation(scale * self.linear(x) + shift)

    def forward_lowrank_gate(self, x, u_mat, v_mat):
        """lowrank gate modulation

        Args:
            x (torch.Tensor): input [batch_size, ..., dim_in]
            u_mat (torch.Tensor): U matrix [batch_size, rank, dim_out]
            v_mat (torch.Tensor): V matrix [batch_size, rank, dim_in]

        Returns:
            torch.Tensor: activation((sigmoid(UV) * W)x + b) [batch_size, ..., dim_out]
        """
        rank = u_mat.shape[1]
        G = einsum(u_mat, v_mat, "b r do, b r di -> b do di") / sqrt(rank)
        G = torch.sigmoid(G)

        modulated_weight = G * self.linear.weight.unsqueeze(0)
        bias = 0.0 if self.linear.bias is None else self.linear.bias
        out = einsum(modulated_weight, x, "b o i, b ... i -> b ... o")
        out = out + bias
        return self.activation(out)


class LowRankINRLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        rank=1,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.linear = nn.Linear(rank, dim_out, bias=use_bias)
        self.activation = nn.ReLU()

    def forward(self, x, v_mat):
        weight = einsum(self.linear.weight, v_mat, "do r, b r di -> b do di")
        x = einsum(weight, x, "b do di, b n di -> b n do")
        return self.activation(x)


class FourierFeatureINRLayerFirst(INRLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
        ff_dim=128,
    ):
        super().__init__(
            dim_in,
            dim_out,
            use_bias,
        )

        assert ff_dim % 2 == 0 and ff_dim > 0

        self.linear = nn.Linear(dim_in * ff_dim // 2, dim_out, bias=use_bias)
        self.convert_pos_emb = FixedFourierFeatures(ff_dim // 2)

    def forward(self, x, scale=1.0, shift=0.0):
        x = self.convert_pos_emb(x)
        return self.activation(scale * self.linear(x) + shift)


class RandomFourierFeatureINRLayerFirst(INRLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
        ff_dim=128,
        sigma=10,
    ):
        super().__init__(
            dim_in,
            dim_out,
            use_bias,
        )

        assert ff_dim % 2 == 0 and ff_dim > 0

        self.linear = nn.Linear(ff_dim, dim_out, bias=use_bias)
        self.convert_pos_emb = RandomFourierFeatures(ff_dim, dim_in, sigma)

    def forward(self, x, scale=1.0, shift=0.0):
        x = self.convert_pos_emb(x)
        return self.activation(scale * self.linear(x) + shift)


class INRModule(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.encoder = None
        self.body = nn.ModuleList()

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.body:
            x = layer(x)

        x = self.decoder(x)
        return x

    def shift_modulated_forward(self, x0, modulations, modulate_first=False):
        if modulations is None:
            return self(x0)

        """
        Shape of x0
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shpae of modulations
        1D - (bsz, nl * dim_hidden, mss)
        2D - (bsz, nl * dim_hidden, mss, mss)
        3D - (bsz, nl * dim_hidden, mss, mss, mss)
        """

        Dx = x0.shape[-1]
        bsz, D, mss = modulations.shape[:3]
        nl = D // self.dim_hidden

        l = 0
        if mss == 1:
            # no spatial functa
            shifts = rearrange(modulations, "b (nl dim) ... -> nl b 1 (dim ...)", nl=nl)  # (nl, bsz, 1, dim_hidden)

        else:
            # spatial functa
            shifts = grid_sample(modulations, x0)  # (bsz, num_points, nl * dim_hidden)
            shifts = rearrange(shifts, "b n (nl dim) -> nl b n dim", nl=nl)  # (nl, bsz, num_points, dim_hidden)

        if modulate_first:
            x = self.encoder(x0, shift=shifts[l])
            l += 1
        else:
            x = self.encoder(x0)

        for layer in self.body:
            if l < nl:
                x = layer(x, shift=shifts[l])
                l += 1
            else:
                x = layer(x)

        x = self.decoder(x)
        return x

    def scale_modulated_forward(self, x0, modulations, modulate_first=False):
        if modulations is None:
            return self(x0)

        """
        Shape of x0
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shpae of modulations
        1D - (bsz, nl * dim_hidden, mss)
        2D - (bsz, nl * dim_hidden, mss, mss)
        3D - (bsz, nl * dim_hidden, mss, mss, mss)
        """

        Dx = x0.shape[-1]
        bsz, D, mss = modulations.shape[:3]
        nl = D // self.dim_hidden

        l = 0
        if mss == 1:
            # no spatial functa
            scales = rearrange(modulations, "b (nl dim) ... -> nl b 1 (dim ...)", nl=nl)  # (nl, bsz, 1, dim_hidden)

        else:
            # spatial functa
            scales = grid_sample(modulations, x0)  # (bsz, num_points, nl * dim_hidden)
            scales = rearrange(scales, "b n (nl dim) -> nl b n dim", nl=nl)  # (nl, bsz, num_points, dim_hidden)

        if modulate_first:
            x = self.encoder(x0, scale=scales[l])
            l += 1
        else:
            x = self.encoder(x0)

        for layer in self.body:
            if l < nl:
                x = layer(x, scale=scales[l])
                l += 1
            else:
                x = layer(x)

        x = self.decoder(x)
        return x

    def scaleshift_modulated_forward(self, x0, modulations, modulate_first=False):
        if modulations is None:
            return self(x0)

        """
        Shape of x0
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shpae of modulations
        1D - (bsz, nl * dim_hidden, mss)
        2D - (bsz, nl * dim_hidden, mss, mss)
        3D - (bsz, nl * dim_hidden, mss, mss, mss)
        """

        Dx = x0.shape[-1]
        bsz, D, mss = modulations.shape[:3]
        nl = D // (self.dim_hidden * 2)

        l = 0
        if mss == 1:
            # no spatial functa
            scaleshifts = rearrange(
                modulations, "b (nl p dim) ... -> nl p b 1 (dim ...)", nl=nl, p=2
            )  # (nl, 2, bsz, 1, dim_hidden)

        else:
            # spatial functa
            scaleshifts = grid_sample(modulations, x0)  # (bsz, num_points, nl * dim_hidden)
            scaleshifts = rearrange(
                scaleshifts, "b n (nl p dim) -> nl p b n dim", nl=nl, p=2
            )  # (nl, p, bsz, num_points, dim_hidden)

        if modulate_first:
            x = self.encoder(x0, scale=scaleshifts[l, 0], shift=scaleshifts[l, 1])
            l += 1
        else:
            x = self.encoder(x0)

        for layer in self.body:
            if l < nl:
                x = layer(x, scale=scaleshifts[l, 0], shift=scaleshifts[l, 1])
                l += 1
            else:
                x = layer(x)

        x = self.decoder(x)
        return x


class BasicINR(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(
            dim_in,
            dim_out,
            dim_hidden,
            num_layers,
        )

        self.encoder = INRLayer(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            use_bias=use_bias,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                INRLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class Siren(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(dim_in, dim_out, dim_hidden, num_layers)

        self.encoder = SirenLayerFirst(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            w0=w0_initial,
            w0_learnable=False,
            use_bias=use_bias,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                SirenLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0,
                    w0_learnable=False,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class FourierFeatureINR(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        ff_dim,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(
            dim_in,
            dim_out,
            dim_hidden,
            num_layers,
        )

        self.encoder = FourierFeatureINRLayerFirst(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            use_bias=use_bias,
            ff_dim=ff_dim,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                INRLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class RandomFourierFeatureINR(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        ff_dim,
        sigma=10.0,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(
            dim_in,
            dim_out,
            dim_hidden,
            num_layers,
        )

        self.encoder = RandomFourierFeatureINRLayerFirst(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            use_bias=use_bias,
            ff_dim=ff_dim,
            sigma=sigma,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                INRLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class LowRankINR(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        rank,
        ff_dim,
        sigma=10.0,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.dim_in = dim_in
        self.dim_out = dim_out

        if sigma == 0.0:
            self.encoder = INRLayer(
                dim_in=self.dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
            )
        else:
            self.encoder = RandomFourierFeatureINRLayerFirst(
                dim_in=self.dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                ff_dim=ff_dim,
                sigma=sigma,
            )

        self.body = nn.ModuleList()
        for i in range(num_layers - 2):
            if i == 0:
                self.body.append(
                    LowRankINRLayer(
                        dim_in=dim_hidden,
                        dim_out=dim_hidden,
                        rank=rank,
                        use_bias=use_bias,
                    )
                )
            else:
                self.body.append(
                    INRLayer(
                        dim_in=dim_hidden,
                        dim_out=dim_hidden,
                        use_bias=use_bias,
                    )
                )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.body:
            x = layer(x)

        x = self.decoder(x)
        return x

    def lowrank_modulated_forward(self, x, modulation):
        if modulation is None:
            return self(x)

        """
        Shape of x
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shape of modulation
        1D - (bsz, n, dim_hidden)
        2D - (bsz, n, dim_hidden)
        3D - (bsz, n, dim_hidden)
        """

        x = self.encoder(x)

        # apply modulation
        for i, layer in enumerate(self.body):
            if i == 0:
                x = layer(x, modulation)
            else:
                x = layer(x)

        x = self.decoder(x)

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, einsum
from einops.layers.torch import Rearrange


class MultimodalMetaModel(nn.Module):
    def __init__(
        self,
        args,
        modes,
        latent_spatial_shapes,
        latent_dims,
        inr_dict,
        grad_encoder_dict,
        meta_sgd_dict,
    ) -> None:
        super().__init__()

        self.args = args
        self.modes = modes
        self.num_modes = len(modes)
        self.dim_hidden = inr_dict["dim_hidden"]
        self.num_layers = inr_dict["num_layers"]

        self.inr_type = inr_dict["inr_type"]
        self.inr_dict = inr_dict
        self.modulate_scale = inr_dict["modulate_scale"]
        self.modulate_shift = inr_dict["modulate_shift"]
        self.modulate_first = inr_dict["modulate_first"]
        if self.modulate_first:
            self.num_modulate_layers = inr_dict["num_layers"] - 1
        else:
            self.num_modulate_layers = inr_dict["num_layers"] - 2
        self.latent_spatial_shapes = latent_spatial_shapes

        if "composer" in self.inr_type:
            self.latent_dims = {mode: self.dim_hidden for mode in modes}  # if latent_dims is None else latent_dims
            latent_dims = None
        else:
            self.latent_dims = latent_dims

        self.grad_encoder_dict = grad_encoder_dict
        self.meta_sgd_dict = meta_sgd_dict

        # Init INR layers
        # ---------------------------------------------------------------------------------------

        self.dims_in = get_input_dims(modes)
        self.dims_out = get_output_dims(modes)

        self.inr = nn.ModuleDict()

        if self.inr_type == "siren":
            for mode in self.modes:
                self.inr[mode] = Siren(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    w0=inr_dict["w0"],
                    w0_initial=inr_dict["w0_initial"],
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "rffn":
            for mode in self.modes:
                self.inr[mode] = RandomFourierFeatureINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    ff_dim=inr_dict["ff_dim"],
                    sigma=inr_dict["sigma"],
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "ffn":
            for mode in self.modes:
                self.inr[mode] = FourierFeatureINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    ff_dim=inr_dict["ff_dim"],
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "basic":
            for mode in self.modes:
                self.inr[mode] = BasicINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "composer":
            for mode in self.modes:
                self.inr[mode] = LowRankINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    rank=latent_spatial_shapes[mode],
                    ff_dim=inr_dict["ff_dim"],
                    sigma=inr_dict["sigma"],
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )

        self.latent_shapes = dict()

        if "composer" in self.inr_type:
            self.modulate_dim = self.dim_hidden
        else:
            if self.modulate_scale and self.modulate_shift:
                self.modulate_dim = self.dim_hidden * 2 * self.num_modulate_layers
            else:
                self.modulate_dim = self.dim_hidden * self.num_modulate_layers

        for mode in self.modes:
            if "composer" in self.inr_type:
                self.latent_shapes[mode] = latent_spatial_shapes[mode]
            else:
                self.latent_shapes[mode] = latent_spatial_shapes[mode] ** self.dims_in[mode]
                if "era5" == self.args.dataset_config["name"]:
                    # ERA5 needs 1:2 grid
                    self.latent_shapes[mode] *= 2

        self.latent_prior_embeds = nn.ParameterDict()

        for mode in self.modes:
            self.latent_prior_embeds[mode] = nn.Parameter(
                torch.randn(1, self.latent_shapes[mode], self.latent_dims[mode]) * 0.2, requires_grad=True
            )

        self.latent_to_modulate = nn.ModuleDict()

        if "composer" in self.inr_type:
            # Composers for INRs
            for mode in self.modes:
                self.latent_to_modulate[mode] = (
                    nn.Sequential(
                        nn.LayerNorm(self.latent_dims[mode]),
                        nn.Linear(self.latent_dims[mode], self.modulate_dim),
                    )
                    if latent_dims is not None
                    else nn.Identity()
                )
        else:
            # Functa for INRs
            for mode in self.modes:
                if self.dims_in[mode] == 1:  # 1D data
                    self.latent_to_modulate[mode] = nn.Sequential(
                        nn.LayerNorm(self.latent_dims[mode]),  # (bsz, mss, D)
                        LatentReshape1D(),  # (bsz, D, mss)
                        nn.Conv1d(self.latent_dims[mode], self.modulate_dim, 3, 1, 1),  # (bsz, D, mss)
                    )
                elif self.dims_in[mode] == 2:  # 2D data
                    self.latent_to_modulate[mode] = nn.Sequential(
                        nn.LayerNorm(self.latent_dims[mode]),  # (bsz, mss * mss, D)
                        LatentReshape2D(),  # (bsz, D, mss, mss)
                        nn.Conv2d(self.latent_dims[mode], self.modulate_dim, 3, 1, 1),  # (bsz, D, mss, mss)
                    )

        # meta sgd
        # ---------------------------------------------------------------------------------------

        lr_init = meta_sgd_dict["inner_lr_init"]
        self.meta_lr = nn.ParameterDict()

        for mode in self.modes:
            if self.args.use_alfa:
                self.meta_lr[mode] = nn.Parameter(
                    torch.zeros(1, self.latent_shapes[mode], 1) + lr_init,
                    requires_grad=meta_sgd_dict["use_meta_sgd"],
                )
            else:
                self.meta_lr[mode] = nn.Parameter(
                    torch.zeros(1, self.latent_shapes[mode], self.latent_dims[mode]) + lr_init,
                    requires_grad=meta_sgd_dict["use_meta_sgd"],
                )
            if lr_init == 0:
                nn.init.uniform_(self.meta_lr[mode], 0.005, 1.0)

        # uncertainty-based loss weighting
        if self.args.loss_weight_mode == "uncertainty":
            self.logvars = nn.ParameterDict()

            for mode in self.modes:
                self.logvars[mode] = nn.Parameter(
                    torch.zeros(
                        1,
                    )
                    + self.args.logvar_init[mode],
                    requires_grad=True,
                )

        self.use_gap = self.args.use_gap
        self.use_alfa = self.args.use_alfa
        self.use_grad_encoder = grad_encoder_dict["um_depth"] + grad_encoder_dict["mm_depth"] > 0

        # Define State Fusion Transformers (SFTs)
        # ---------------------------------------------------------------------------------------
        if self.use_grad_encoder:
            self.grad_encoder_projection_mlp = nn.ModuleDict()
            self.grad_encoder_state_to_gradient = nn.ModuleDict()
            self.grad_encoder_pos_embed_type = grad_encoder_dict["pos_embed_type"]
            self.grad_encoder_fuser = nn.ModuleDict() if grad_encoder_dict["use_fuser"] else False
            self.log_grad_scaler = nn.ParameterDict()
            grad_scaler_init = grad_encoder_dict["grad_scaler_init"] if "grad_scaler_init" in grad_encoder_dict else 1.0

            if grad_encoder_dict["use_grad"]:
                self.grad_encoder_grad_ln = nn.ModuleDict()

            if grad_encoder_dict["use_latent"]:
                self.grad_encoder_latent_ln = nn.ModuleDict()

            for mode in self.modes:
                self.log_grad_scaler[mode] = nn.Parameter(
                    torch.tensor(grad_scaler_init).log(),
                    requires_grad=self.args.grad_encoder_grad_scaler_learnable,
                )

                num_channels = 0
                if grad_encoder_dict["use_grad"]:
                    num_channels += self.latent_dims[mode]
                    self.grad_encoder_grad_ln[mode] = nn.LayerNorm(self.latent_dims[mode], eps=1e-5)

                if grad_encoder_dict["use_latent"]:
                    num_channels += self.latent_dims[mode]
                    self.grad_encoder_latent_ln[mode] = nn.LayerNorm(self.latent_dims[mode], eps=1e-5)

                assert num_channels > 0

                self.grad_encoder_projection_mlp[mode] = nn.Sequential(
                    Mlp(
                        in_features=num_channels,
                        hidden_features=grad_encoder_dict["dim"],
                        out_features=grad_encoder_dict["dim"],
                        depth=grad_encoder_dict["projection_mlp_depth"],
                    ),
                )

                if grad_encoder_dict["use_fuser"]:
                    num_states = 1
                    if grad_encoder_dict["um_depth"] > 0:
                        num_states += 1
                    if grad_encoder_dict["mm_depth"] > 0:
                        num_states += 1
                    self.grad_encoder_fuser[mode] = nn.Sequential(
                        Rearrange("b n d -> b d n"),
                        nn.GroupNorm(
                            num_groups=num_states,
                            num_channels=grad_encoder_dict["dim"] * num_states,
                        ),
                        Rearrange("b d n -> b n d"),
                        Mlp(
                            in_features=grad_encoder_dict["dim"] * num_states,
                            hidden_features=grad_encoder_dict["dim"],
                            out_features=grad_encoder_dict["dim"],
                            depth=grad_encoder_dict["depth_fuser"],
                        ),
                        nn.LayerNorm(grad_encoder_dict["dim"]),
                    )

                self.grad_encoder_state_to_gradient[mode] = nn.Linear(
                    grad_encoder_dict["dim"],
                    self.latent_dims[mode],
                )

                self.grad_encoder_pos_embeds = nn.ParameterDict()
                if self.grad_encoder_pos_embed_type in ["fixed"]:
                    for mode in self.modes:
                        self.grad_encoder_pos_embeds[mode] = self._create_fourier_embeds(grad_encoder_dict["dim"], mode)
                elif self.grad_encoder_pos_embed_type in ["learned"]:
                    for mode in self.modes:
                        self.grad_encoder_pos_embeds[mode] = nn.Parameter(
                            torch.randn(self.latent_shapes[mode], grad_encoder_dict["dim"]) * 0.2,
                            requires_grad=True,
                        )

            self.grad_encoder_um = nn.ModuleDict()
            for mode in self.modes:
                self.grad_encoder_um[mode] = Transformer(
                    dim=grad_encoder_dict["dim"],
                    depth=grad_encoder_dict["um_depth"],
                    heads=grad_encoder_dict["heads"],
                    dim_head=grad_encoder_dict["dim_head"],
                    mlp_dim=int(grad_encoder_dict["dim"] * grad_encoder_dict["mlp_ratio"]),
                    dropout=grad_encoder_dict["dropout"],
                )
            self.grad_encoder_mm = Transformer(
                dim=grad_encoder_dict["dim"],
                depth=grad_encoder_dict["mm_depth"],
                heads=grad_encoder_dict["heads"],
                dim_head=grad_encoder_dict["dim_head"],
                mlp_dim=int(grad_encoder_dict["dim"] * grad_encoder_dict["mlp_ratio"]),
                dropout=grad_encoder_dict["dropout"],
            )

        if self.args.use_alfa:
            self.alfa = nn.ModuleDict()
            self.beta_init_dict = nn.ParameterDict()
            for mode in self.modes:
                input_dim = self.latent_shapes[mode] * 2
                if self.args.dim_alfa > 0:
                    hidden_dim = self.args.dim_alfa
                else:
                    hidden_dim = input_dim

                self.alfa[mode] = Mlp(
                    in_features=input_dim,
                    hidden_features=hidden_dim,
                    out_features=input_dim,
                    depth=self.args.depth_alfa,
                )
                self.beta_init_dict[mode] = nn.Parameter(
                    torch.ones(1, self.latent_shapes[mode], 1),
                    requires_grad=meta_sgd_dict["use_meta_sgd"],
                )

        self.use_gap = self.args.use_gap
        if self.use_gap:
            self.M = nn.ParameterDict()
            for mode in self.modes:
                shape = min(self.latent_shapes[mode], self.latent_dims[mode])
                self.M[mode] = nn.Parameter(0.928 * torch.ones(shape), requires_grad=True)

    def get_inr_params(self):
        non_inr_keywords = ["grad_enc", "logvars"]

        params = {}
        for k, v in dict(self.named_parameters()).items():
            non_inr_keywords_exist = []
            for non_inr_keyword in non_inr_keywords:
                non_inr_keywords_exist += [non_inr_keyword in k]
            if sum(non_inr_keywords_exist) == 0:
                # print('inr_params:', k)
                params[k] = v
        return params

    def get_non_inr_params(self):
        non_inr_keywords = ["grad_enc"]

        params = {}
        for k, v in dict(self.named_parameters()).items():
            non_inr_keywords_exist = []
            for non_inr_keyword in non_inr_keywords:
                non_inr_keywords_exist += [non_inr_keyword in k]
            if sum(non_inr_keywords_exist) > 0:
                # print('non_inr_params:', k)
                params[k] = v
        return params

    def get_logvars(self):
        non_inr_keywords = ["logvars"]

        params = {}
        for k, v in dict(self.named_parameters()).items():
            non_inr_keywords_exist = []
            for non_inr_keyword in non_inr_keywords:
                non_inr_keywords_exist += [non_inr_keyword in k]
            if sum(non_inr_keywords_exist) > 0:
                params[k] = v
        return params

    def get_parameters(self, keys=None):
        if keys is None:
            params = [v for k, v in self.named_parameters()]
        else:
            if isinstance(keys, (list, tuple)):
                params = [v for k, v in self.named_parameters() if len([key for key in keys if key in k]) > 0]
            elif isinstance(keys, str):
                params = [v for k, v in self.named_parameters() if keys in k]

        return params

    def _create_fourier_embeds(self, dim, mode):
        w = torch.exp(torch.linspace(0, 8, dim // 2 // self.dims_in[mode]))
        coords = []
        input_range = get_input_range(mode)
        for dim_idx in range(self.dims_in[mode]):
            if self.latent_spatial_shapes[mode] == 1:
                coords.append(torch.linspace(-0, 0, self.latent_spatial_shapes[mode]))
            else:
                if "era5" == self.args.dataset_config["name"]:
                    # ERA5 needs 1:2 grid
                    coords.append(
                        torch.linspace(-input_range, input_range, self.latent_spatial_shapes[mode] * (dim_idx + 1))
                    )
                else:
                    coords.append(torch.linspace(-input_range, input_range, self.latent_spatial_shapes[mode]))

        coords = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
        coords = einsum(coords, w, "... d, fdim -> ... d fdim").view(*coords.shape[:-1], -1)
        coords = torch.cat([torch.cos(torch.pi * coords), torch.sin(torch.pi * coords)], dim=-1)
        coords = coords.reshape(1, -1, dim)
        coords = nn.Parameter(coords, requires_grad=False)
        return coords

    def init_latent(self, batch_size):
        latent_prior_dict = {}
        for mode in self.modes:
            latent_prior_dict[mode] = repeat(self.latent_prior_embeds[mode], "1 ... -> bsz ...", bsz=batch_size)

        return latent_prior_dict

    def fuse_states(self, grad_dict, latent_dict):
        if not self.use_grad_encoder and not self.use_alfa and not self.use_gap:
            return grad_dict, latent_dict

        elif self.use_grad_encoder:
            ordinal_state_dict = {}
            unimodal_state_dict = {}
            multimodal_state_dict = {}
            modified_grad_dict = {}

            n_tokens = [0]
            states = []
            for mode in self.modes:
                B, N, D = grad_dict[mode].shape

                input_features = []
                if self.grad_encoder_dict["use_grad"]:
                    input_features += [self.grad_encoder_grad_ln[mode](grad_dict[mode])]
                if self.grad_encoder_dict["use_latent"]:
                    input_features += [self.grad_encoder_latent_ln[mode](latent_dict[mode])]

                input_features = torch.cat(input_features, dim=-1)
                input_features = self.grad_encoder_projection_mlp[mode](input_features)

                state = input_features
                if self.grad_encoder_fuser:
                    ordinal_state_dict[mode] = state

                if self.use_grad_encoder:
                    state += self.grad_encoder_pos_embeds[mode]

                state = self.grad_encoder_um[mode](state)

                if self.grad_encoder_fuser:
                    unimodal_state_dict[mode] = state

                n_tokens += [n_tokens[-1] + state.shape[1]]

                states += [state]

            states = torch.cat(states, dim=1)
            states = self.grad_encoder_mm(states)

            for i, mode in enumerate(self.modes):
                state = states[:, n_tokens[i] : n_tokens[i + 1], :]
                multimodal_state_dict[mode] = state

            for mode in self.modes:
                if self.grad_encoder_fuser:
                    state_features = [ordinal_state_dict[mode]]
                    if self.grad_encoder_dict["um_depth"] > 0:
                        state_features += [unimodal_state_dict[mode]]
                    if self.grad_encoder_dict["mm_depth"] > 0:
                        state_features += [multimodal_state_dict[mode]]
                    state_features = torch.cat(state_features, dim=-1)
                    state_features = self.grad_encoder_fuser[mode](state_features)
                    modified_grad_dict[mode] = self.grad_encoder_state_to_gradient[mode](state_features)
                else:
                    modified_grad_dict[mode] = self.grad_encoder_state_to_gradient[mode](multimodal_state_dict[mode])

            return modified_grad_dict, latent_dict

        elif self.use_alfa:
            modified_latent_dict = {}
            modified_grad_dict = {}
            for mode in self.modes:
                states = torch.cat([grad_dict[mode].mean(-1), latent_dict[mode].mean(-1)], -1).flatten(1)
                states = self.alfa[mode](states)
                states = states.reshape(states.shape[0], -1, 1, 2)
                beta, alpha = states[..., 0], states[..., 1]
                modified_latent_dict[mode] = self.beta_init_dict[mode] * beta * latent_dict[mode]
                modified_grad_dict[mode] = alpha * grad_dict[mode]

            return modified_grad_dict, modified_latent_dict

        elif self.use_gap:
            modified_grad_dict = {}
            for mode in self.modes:
                M = self.M[mode]
                M = repeat(torch.diag(F.softplus(M, beta=2)), "n m -> b n m", b=grad_dict[mode].shape[0])
                if self.args.use_gap_approx:
                    if grad_dict[mode].shape[1] >= grad_dict[mode].shape[2]:
                        modified_grad_dict[mode] = grad_dict[mode] @ M
                    else:
                        modified_grad_dict[mode] = M @ grad_dict[mode]
                else:
                    u, _, _ = torch.svd(grad_dict[mode].detach())
                    preconditioner = u @ (M @ u.transpose(2, 1))
                    modified_grad_dict[mode] = preconditioner @ grad_dict[mode]

            return modified_grad_dict, latent_dict

    def get_grad_scale(self, mode):
        if self.grad_encoder_dict["use_grad_scaler"]:
            return self.log_grad_scaler[mode].exp()
        else:
            return 1

    def modulated_forward_single(self, x, latent, mode):
        # 1D - (bsz, lss, ld) -> (bsz, D, lss)
        # 2D - (bsz, lss * lss, ld) -> (bsz, D, lss, lss)
        modulations = self.latent_to_modulate[mode](latent)

        if "composer" in self.inr_type:
            x = self.inr[mode].lowrank_modulated_forward(x, modulations)
        elif self.modulate_scale and self.modulate_shift:
            x = self.inr[mode].scaleshift_modulated_forward(x, modulations, self.modulate_first)
        elif self.modulate_scale:
            x = self.inr[mode].scale_modulated_forward(x, modulations, self.modulate_first)
        elif self.modulate_shift:
            x = self.inr[mode].shift_modulated_forward(x, modulations, self.modulate_first)
        else:
            x = self.inr[mode](x)

        return x