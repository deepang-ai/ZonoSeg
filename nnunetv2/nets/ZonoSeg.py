import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.blocks.dynunet_block import (
    UnetOutBlock,
    UnetResBlock,
    UnetUpBlock,
    get_conv_layer,
)
from monai.networks.blocks.squeeze_and_excitation import ResidualSELayer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.dynunet import DynUNet
from nnunetv2.nets.zonoseg_kernel import ScharrEdge, MultiScaleGaussian


class ZonoSegBlock(nn.Module):
    def __init__(self, dim, out_dim, norm_layer, act_layer):
        super().__init__()
        self.gaussian = MultiScaleGaussian(dim, sizes=[3, 5, 7], sigmas=[0.8, 1.2, 1.4])
        self.scharr = ScharrEdge(dim)
        self.norm = get_norm_layer(norm_layer, spatial_dims=3, channels=dim * 3)
        self.act = get_act_layer(act_layer)
        self.eca = ResidualSELayer(3, dim * 3, acti_type_2="sigmoid")

        self.fuse = get_conv_layer(
            3, dim * 3, out_dim, kernel_size=1, act=act_layer, norm=norm_layer
        )

    def forward(self, x):
        g = self.gaussian(x)
        e = self.scharr(x)
        fuse = self.norm(self.act(torch.cat([x, g, e], dim=1)))
        out = self.eca(fuse)
        return self.fuse(out)




class SparseAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attn_stride=4, num_heads=8):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=attn_stride)
        self.query = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv_trans = nn.ConvTranspose3d(
            out_channels,
            out_channels,
            kernel_size=attn_stride,
            stride=attn_stride,
            groups=out_channels,
        )

    def forward(self, x):
        res = x
        x = self.sparse_sampler(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        B, C, W, H, D = x.size()
        q = q.view(B, self.num_heads, -1, W * H * D).permute(0, 1, 3, 2).contiguous()
        k = k.view(B, self.num_heads, -1, W * H * D).permute(0, 1, 3, 2).contiguous()
        v = v.view(B, self.num_heads, -1, W * H * D).permute(0, 1, 3, 2).contiguous()
        context = F.scaled_dot_product_attention(
            F.normalize(q, dim=-1), F.normalize(k, dim=-1), v
        )

        context = context.permute(0, 1, 3, 2).contiguous().view(B, C, W, H, D)
        context = self.conv_trans(context)
        return context + res


class ZADHead(nn.Module):
    def __init__(self, top_c, bottom_c, num_classes, act_layer, norm_layer):
        super().__init__()

        self.bottom_conv = nn.Conv3d(bottom_c, 1, 3, 1, 1)

        self.ff_conv = nn.Conv3d(
            in_channels=top_c,
            out_channels=top_c,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=top_c,
        )
        self.bf_conv = nn.Conv3d(
            in_channels=top_c,
            out_channels=top_c,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=top_c,
        )
        self.cross_attention = CrossAttentionBlock(
            hidden_size=top_c,
            num_heads=4,
            dropout_rate=0.0,
            use_flash_attention=True,
        )

        self.downsample_factor = 8
        self.conv_trans = nn.ConvTranspose3d(
            top_c,
            top_c,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
            groups=top_c,
        )

        self.se = ResidualSELayer(3, top_c * 3, acti_type_2="sigmoid")

        self.seg_head = UnetOutBlock(3, top_c * 3, num_classes)

    def forward(self, feat, bottom):
        B, C, D, H, W = feat.size()
        bottom = F.sigmoid(
            F.interpolate(
                self.bottom_conv(bottom),
                size=(D, H, W),
                mode="trilinear",
                align_corners=True,
            )
        )

        ff_feat = self.ff_conv(feat * bottom)
        bf_feat = self.bf_conv(feat * (1 - bottom))

        ff_feat_down = F.avg_pool3d(
            ff_feat, kernel_size=1, stride=self.downsample_factor
        )
        bf_feat_down = F.avg_pool3d(
            bf_feat, kernel_size=1, stride=self.downsample_factor
        )

        ff_feat_seq = ff_feat_down.view(B, C, -1).transpose(1, 2)
        bf_feat_seq = bf_feat_down.view(B, C, -1).transpose(1, 2)

        ff_enhanced_seq = self.cross_attention(ff_feat_seq, bf_feat_seq)
        bf_enhanced_seq = self.cross_attention(bf_feat_seq, ff_feat_seq)

        target_d, target_h, target_w = (
            D // self.downsample_factor,
            H // self.downsample_factor,
            W // self.downsample_factor,
        )
        ff_enhanced_down = ff_enhanced_seq.transpose(1, 2).view(
            B, C, target_d, target_h, target_w
        )
        bf_enhanced_down = bf_enhanced_seq.transpose(1, 2).view(
            B, C, target_d, target_h, target_w
        )

        ff_enhanced = self.conv_trans(ff_enhanced_down) + ff_feat
        bf_enhanced = self.conv_trans(bf_enhanced_down) + bf_feat

        fusion = torch.cat((ff_enhanced, bf_enhanced, feat), dim=1)

        seg_res = self.seg_head(self.se(fusion))
        return seg_res




class ZonoSeg(nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        dim_in=None,
        num_classes=None,
        depths=[1, 2, 4, 2],
        embed_dims=[24, 48, 96, 192],
        drop=0.0,
        norm_layer=("instance", {"affine": True}),
        act_layer="GELU",
    ):
        super().__init__()
        if in_channels is not None:
            dim_in = in_channels
        if out_channels is not None:
            num_classes = out_channels
        dim_in = 1 if dim_in is None else dim_in
        num_classes = 2 if num_classes is None else num_classes
        self.num_classes = num_classes
        self.skip = get_conv_layer(
            3,
            dim_in,
            embed_dims[0],
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            dropout=drop,
            conv_only=False,
            norm=norm_layer,
            act=act_layer,
        )
        self.stage0 = get_conv_layer(
            3,
            dim_in,
            embed_dims[0],
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            dropout=drop,
            conv_only=False,
            norm=norm_layer,
            act=act_layer,
        )
        self.skip_down = get_conv_layer(
            3,
            embed_dims[0],
            embed_dims[3],
            kernel_size=(4, 8, 8),
            stride=(4, 8, 8),
            dropout=drop,
            conv_only=False,
            norm=norm_layer,
            act=act_layer,
        )

        self.stages = nn.ModuleList()
        self.downlayers = nn.ModuleList()
        self.translayersdown = nn.ModuleList()
        self.translayersup = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.translayersdown.append(
            SparseAttention(
                in_channels=embed_dims[0],
                out_channels=embed_dims[0],
                num_heads=4,
                attn_stride=2,
            ),
        )
        self.translayersup.append(
            SparseAttention(
                in_channels=embed_dims[3],
                out_channels=embed_dims[3],
                num_heads=4,
                attn_stride=2,
            )
        )

        n = len(depths)
        for i in range(n):
            self.stages.append(
                nn.Sequential(
                    *[
                        UnetResBlock(
                            spatial_dims=3,
                            in_channels=embed_dims[i],
                            out_channels=embed_dims[i],
                            kernel_size=3,
                            stride=1,
                            norm_name=norm_layer,
                            act_name=act_layer,
                        )
                        for _ in range(depths[i])
                    ]
                )
            )

            if i < n - 1:
                k_down = s_down = (1, 2, 2) if i == 0 else (2, 2, 2)
                self.downlayers.append(
                    get_conv_layer(
                        3,
                        embed_dims[i],
                        embed_dims[i + 1],
                        kernel_size=k_down,
                        stride=s_down,
                        dropout=drop,
                        conv_only=False,
                        norm=norm_layer,
                        act=act_layer,
                    )
                )

                j = i + 1
                down_size = (1, 2, 2) if j == 1 else (2, 2, 2)
                self.translayersdown.append(
                    nn.Sequential(
                        get_conv_layer(
                            3,
                            embed_dims[j - 1],
                            embed_dims[j],
                            kernel_size=down_size,
                            stride=down_size,
                            dropout=drop,
                            conv_only=False,
                            norm=norm_layer,
                            act=act_layer,
                        ),
                        SparseAttention(
                            in_channels=embed_dims[j],
                            out_channels=embed_dims[j],
                            num_heads=4,
                            attn_stride=2,
                        ),
                    )
                )

                up_size = (1, 2, 2) if j == n - 1 else (2, 2, 2)
                self.translayersup.append(
                    get_conv_layer(
                        spatial_dims=3,
                        in_channels=embed_dims[n - j],
                        out_channels=embed_dims[n - j - 1],
                        kernel_size=up_size,
                        stride=up_size,
                        norm=norm_layer,
                        act=act_layer,
                        conv_only=False,
                        is_transposed=True,
                    )
                )
            self.fusion.append(
                ZonoSegBlock(
                    dim=3 * embed_dims[i],
                    out_dim=embed_dims[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
            )

        self.up3 = UnetUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[3],
            out_channels=embed_dims[2],
            kernel_size=3,
            stride=2,
            upsample_kernel_size=2,
            norm_name=norm_layer,
            act_name=act_layer,
        )
        self.up2 = UnetUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[2],
            out_channels=embed_dims[1],
            kernel_size=3,
            stride=2,
            upsample_kernel_size=2,
            norm_name=norm_layer,
            act_name=act_layer,
        )
        self.up1 = UnetUpBlock(
            spatial_dims=3,
            in_channels=embed_dims[1],
            out_channels=embed_dims[0],
            kernel_size=3,
            stride=(1, 2, 2),
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_layer,
            act_name=act_layer,
        )
        self.output = ZADHead(
            top_c=embed_dims[0],
            bottom_c=2 * embed_dims[0],
            act_layer=act_layer,
            norm_layer=norm_layer,
            num_classes=num_classes,
        )
        self.apply(DynUNet.initialize_weights)

    def forward(self, x):
        skip_in = self.skip(x)
        x0 = self.stage0(x)

        x_feats = []
        x = x0
        for i in range(len(self.stages)):
            if i > 0:
                x = self.downlayers[i - 1](x)
            x = self.stages[i](x)
            x_feats.append(x)

        x1, x2, x3, x4 = x_feats

        y_down = [None] * len(self.translayersdown)
        y_down[0] = self.translayersdown[0](x1)
        for i in range(1, len(self.translayersdown)):
            y_down[i] = self.translayersdown[i](y_down[i - 1])
        y1down, y2down, y3down, y4down = y_down

        skip_down = self.skip_down(x0)
        y_up = [None] * len(self.translayersup)
        y_up[0] = self.translayersup[0](skip_down)
        for i in range(1, len(self.translayersup)):
            y_up[i] = self.translayersup[i](y_up[i - 1])
        y4up, y3up, y2up, y1up = y_up

        x4 = self.fusion[3](torch.cat([x4, y4down, y4up], dim=1))
        x3 = self.fusion[2](torch.cat([x3, y3down, y3up], dim=1))
        x2 = self.fusion[1](torch.cat([x2, y2down, y2up], dim=1))
        x1 = self.fusion[0](torch.cat([x1, y1down, y1up], dim=1))

        out3 = self.up3(x4, x3)
        out2 = self.up2(out3, x2)
        out1 = self.up1(out2, x1)

        return self.output(skip_in, torch.cat([out1, x0], dim=1))


def optimize_for_inference(
    model: nn.Module,
    compile_model: bool = True,
    compile_mode: str = "reduce-overhead",
    dtype: torch.dtype | None = None,
    matmul_precision: str | None = None,
) -> nn.Module:
    """Apply optional inference-only optimizations without changing parameter shapes."""
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
    return model
