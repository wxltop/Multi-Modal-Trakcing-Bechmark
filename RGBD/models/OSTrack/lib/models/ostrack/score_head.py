import torch
from torch import nn
from timm.models.layers import trunc_normal_

from lib.models.ostrack.head import MLP

from lib.models.ostrack.layers.cross_attn import CABlock_


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class ScoreTransformer(nn.Module):
    def __init__(
            self,
            n_cls,
            n_layers,
            d_model,
            d_encoder,
            n_heads,
            patch_size=16,
            mlp_ratio=4.,
            drop_path_rate=0,
            dropout=0,
            box_token=True,
            n_mlp_layers=12,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.Sequential(*[
            CABlock_(
                dim=d_model, num_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=True, drop=0,
                attn_drop=0, drop_path=dpr[i], norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(n_layers)])

        self.box_token = box_token
        if box_token:
            self.cls_proj = nn.Linear(4, d_encoder, bias=True)
        else:
            self.cls_token = nn.Parameter(torch.randn(1, n_cls, d_model))

        self.norm = nn.LayerNorm(d_model)

        self.score_head = MLP(d_model, d_model, 1, n_mlp_layers)

        self.apply(init_weights)
        # trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, search_box=None):
        b, l, c = x.shape

        if not self.box_token:
            cls_tokens = self.cls_token.expand(b, -1, -1)
        else:
            assert search_box is not None
            cls_tokens = self.cls_proj(search_box).unsqueeze(1)

        for blk in self.blocks:
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)

        cls_tokens = self.norm(x)[:, 0]
        out_scores = self.score_head(cls_tokens)  # (b, 1, 1)

        return out_scores
