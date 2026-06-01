import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from .layers import (
    KANBlock, PatchEmbed, ConvLayer, D_ConvLayer
)


class AWEPWaveletTransform(nn.Module):
    """
    
    Haar，
    """
    def __init__(self, wavelet_type='haar', mode='symmetric'):
        super().__init__()
        self.wavelet_type = wavelet_type
        self.mode = mode
        
        if wavelet_type == 'haar':
            self.register_buffer('h0', torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.float32))
            self.register_buffer('h1', torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], dtype=torch.float32))
        
    def dwt_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D
        
        Args:
            x:  [B, C, H, W]
            
        Returns:
            ll:  [B, C, H//2, W//2]
            lh:  [B, C, H//2, W//2]
            hl:  [B, C, H//2, W//2]
            hh:  [B, C, H//2, W//2]
        """
        B, C, H, W = x.shape
        
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            W += 1
            
        x_even = x[:, :, :, 0::2]
        x_odd = x[:, :, :, 1::2]
        
        l = (x_even + x_odd) / np.sqrt(2)
        h = (x_even - x_odd) / np.sqrt(2)
        
        ll_even = l[:, :, 0::2, :]
        ll_odd = l[:, :, 1::2, :]
        ll = (ll_even + ll_odd) / np.sqrt(2)
        
        lh = (ll_even - ll_odd) / np.sqrt(2)
        
        hl_even = h[:, :, 0::2, :]
        hl_odd = h[:, :, 1::2, :]
        hl = (hl_even + hl_odd) / np.sqrt(2)
        
        hh = (hl_even - hl_odd) / np.sqrt(2)
        
        return ll, lh, hl, hh
    
    def idwt_2d(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """
        2D
        
        Args:
            ll:  [B, C, H//2, W//2]
            lh:  [B, C, H//2, W//2]
            hl:  [B, C, H//2, W//2]
            hh:  [B, C, H//2, W//2]
            
        Returns:
             [B, C, H, W]
        """
        B, C, H, W = ll.shape
        
        l_even = (ll + lh) / np.sqrt(2)
        l_odd = (ll - lh) / np.sqrt(2)
        h_even = (hl + hh) / np.sqrt(2)
        h_odd = (hl - hh) / np.sqrt(2)
        
        l = torch.zeros(B, C, H*2, W, device=ll.device, dtype=ll.dtype)
        l[:, :, 0::2, :] = l_even
        l[:, :, 1::2, :] = l_odd
        
        h = torch.zeros(B, C, H*2, W, device=ll.device, dtype=ll.dtype)
        h[:, :, 0::2, :] = h_even
        h[:, :, 1::2, :] = h_odd
        
        x_even = (l + h) / np.sqrt(2)
        x_odd = (l - h) / np.sqrt(2)
        
        x = torch.zeros(B, C, H*2, W*2, device=ll.device, dtype=ll.dtype)
        x[:, :, :, 0::2] = x_even
        x[:, :, :, 1::2] = x_odd
        
        return x


class AWEPDenoising(nn.Module):
    """
    （WaveletDenoising）
    
    ，
    """
    def __init__(self, channels: int = 3, threshold: float = 0.1, soft_threshold: bool = True):
        super().__init__()
        self.channels = channels
        self.threshold = threshold
        self.soft_threshold = soft_threshold
        self.wavelet = AWEPWaveletTransform('haar')
        
        self.adaptive_threshold = nn.Parameter(torch.tensor(threshold))
        
    def soft_thresholding(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """"""
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)
    
    def hard_thresholding(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """"""
        return x * (torch.abs(x) > threshold).float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            x:  [B, C, H, W]
            
        Returns:
             [B, C, H, W]
        """
        original_size = x.shape[-2:]
        
        ll, lh, hl, hh = self.wavelet.dwt_2d(x)
        
        threshold = torch.abs(self.adaptive_threshold)
        
        if self.soft_threshold:
            lh_denoised = self.soft_thresholding(lh, threshold)
            hl_denoised = self.soft_thresholding(hl, threshold)
            hh_denoised = self.soft_thresholding(hh, threshold)
        else:
            lh_denoised = self.hard_thresholding(lh, threshold)
            hl_denoised = self.hard_thresholding(hl, threshold)
            hh_denoised = self.hard_thresholding(hh, threshold)
        
        reconstructed = self.wavelet.idwt_2d(ll, lh_denoised, hl_denoised, hh_denoised)
        
        if reconstructed.shape[-2:] != original_size:
            reconstructed = reconstructed[:, :, :original_size[0], :original_size[1]]
            
        return reconstructed


class AWEP(nn.Module):
    """
    （WaveletPreprocess）
    
    v4：
    ：v3，
    """
    def __init__(self, channels: int = 3, enable_denoising: bool = True, 
                 denoise_threshold: float = 0.08):
        super().__init__()
        self.channels = channels
        self.enable_denoising = enable_denoising
        
        if enable_denoising:
            self.denoising = AWEPDenoising(
                channels=channels, 
                threshold=denoise_threshold,
                soft_threshold=True
            )
        
        self.edge_enhance = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.xavier_uniform_(self.edge_enhance.weight)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            x:  [B, 3, H, W]
            
        Returns:
             [B, 3, H, W] ()
        """
        original = x
        
        if self.enable_denoising:
            denoised = self.denoising(x)
        else:
            denoised = x
            
        enhanced = self.edge_enhance(denoised)
        
        fusion_alpha = torch.sigmoid(self.fusion_weight)
        output = (1 - fusion_alpha) * denoised + fusion_alpha * enhanced
        
        return output


class AWEPFeatureExtractor(nn.Module):
    """
    （WaveletFeatureExtractor）()
    
    ，
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = AWEPWaveletTransform('haar')
        
        self.ll_conv = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.lh_conv = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.hl_conv = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.hh_conv = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            x:  [B, C, H, W]
            
        Returns:
             [B, out_channels, H//2, W//2]
        """
        ll, lh, hl, hh = self.wavelet.dwt_2d(x)
        
        ll_feat = F.relu(self.ll_conv(ll))
        lh_feat = F.relu(self.lh_conv(lh))
        hl_feat = F.relu(self.hl_conv(hl))
        hh_feat = F.relu(self.hh_conv(hh))
        
        combined = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)
        output = self.fusion_conv(combined)
        
        return output


WaveletPreprocess = AWEP
WaveletDenoising = AWEPDenoising
WaveletTransform = AWEPWaveletTransform


class DSFF(nn.Module):
    """
    - (Detail-Semantic Feature Fusion)
    
    ，。
    ：（）（）
    
    :
        channel: 
    """
    def __init__(self, channel):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        """
        
        :
            xs: ，
            anchor: ，
            
        :
        """
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class DSFFChannelAttention(nn.Module):
    """"""
    def __init__(self, in_planes, ratio=16):
        super(DSFFChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DSFFSpatialAttention(nn.Module):
    """"""
    def __init__(self, kernel_size=7):
        super(DSFFSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class DatasetAdaptiveAttention(nn.Module):
    """
    （t1, t2）
    """
    def __init__(self, channels, dataset_type='busi', attention_alpha=0.1):
        super().__init__()
        self.dataset_type = dataset_type.lower()
        self.channels = channels
        
        self.attention_net = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
        if self.dataset_type == 'busi':
            self.activation = nn.Tanh()
            self.alpha = nn.Parameter(torch.tensor(attention_alpha * 0.5))
            self.use_spatial = False
            
        elif self.dataset_type == 'cvc':
            self.activation = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.tensor(attention_alpha))
            self.use_spatial = True
            
        elif self.dataset_type == 'glas':
            self.activation = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.tensor(attention_alpha * 0.3))
            self.use_spatial = False
            
        else:
            self.activation = nn.Sigmoid()
            self.alpha = nn.Parameter(torch.tensor(attention_alpha))
            self.use_spatial = False
        
        if self.use_spatial:
            self.spatial_attention = DSFFSpatialAttentionV2(kernel_size=7)
    
    def forward(self, x):
        channel_att = self.activation(self.attention_net(x))
        
        if self.dataset_type == 'busi':
            enhanced = x + self.alpha * channel_att
        else:
            enhanced = x + self.alpha * (x * channel_att)
        
        if self.use_spatial and hasattr(self, 'spatial_attention'):
            spatial_att = self.spatial_attention(enhanced)
            enhanced = enhanced * spatial_att
        
        return enhanced


class DSFFSpatialAttentionV2(nn.Module):
    """DSFF"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


ASFF = DSFF
ASFFChannelAttention = DSFFChannelAttention
ASFFSpatialAttention = DSFFSpatialAttention
ChannelAttention = DSFFChannelAttention
SpatialAttention = DSFFSpatialAttention


class UKDW(nn.Module):
    """
    UKDW: U-KAN with Detail-Semantic Feature Fusion and Wavelet-based Enhancement
    
    U-KAN + DSFF (-) + AWEP ()
    
    ：
    1. AWEP: 
    2. DSFF: (t1,t2)(t3,t4)
    3. Dataset-Adaptive Attention: 

    :
        num_classes (int): 
        input_channels (int): ，3（RGB）
        deep_supervision (bool): 
        img_size (int): 
        embed_dims (list): 
        no_kan (bool): KAN
        drop_rate (float): dropout
        drop_path_rate (float): drop path
        norm_layer: 
        depths (list): 
        enable_wavelet (bool): AWEP
        wavelet_denoise_threshold (float): 
        detail_dsff_channel (int): DSFF
        semantic_dsff_channel (int): DSFF
        use_detail_attention (bool): 
        attention_alpha (float): 
        dataset_type (str):  ('busi', 'cvc', 'glas')
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], enable_wavelet=True, wavelet_denoise_threshold=0.08,
    detail_dsff_channel=24, semantic_dsff_channel=36,
    use_detail_attention=False, attention_alpha=0.1, dataset_type='busi',
    **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.use_hierarchical_dsff = True
        self.use_detail_attention = use_detail_attention
        self.detail_dsff_channel = detail_dsff_channel
        self.semantic_dsff_channel = semantic_dsff_channel
        self.dataset_type = dataset_type

        self.deep_supervision = deep_supervision
        self.enable_wavelet = enable_wavelet

        if self.enable_wavelet:
            self.awep_preprocess = AWEP(
                channels=input_channels,
                enable_denoising=True,
                denoise_threshold=wavelet_denoise_threshold
            )

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.detail_translayers = nn.ModuleList([
            nn.Conv2d(kan_input_dim//8, self.detail_dsff_channel, kernel_size=1),   # t1
            nn.Conv2d(kan_input_dim//4, self.detail_dsff_channel, kernel_size=1)    # t2
        ])

        self.semantic_translayers = nn.ModuleList([
            nn.Conv2d(kan_input_dim, self.semantic_dsff_channel, kernel_size=1),    # t3
            nn.Conv2d(embed_dims[1], self.semantic_dsff_channel, kernel_size=1)     # t4
        ])

        self.detail_dsff = DSFF(self.detail_dsff_channel)
        self.semantic_dsff = DSFF(self.semantic_dsff_channel)

        self.detail_expand = nn.ModuleList([
            nn.Conv2d(self.detail_dsff_channel, kan_input_dim//8, kernel_size=1),   # 24→32 for t1
            nn.Conv2d(self.detail_dsff_channel, kan_input_dim//4, kernel_size=1)    # 24→64 for t2
        ])

        self.semantic_expand = nn.ModuleList([
            nn.Conv2d(self.semantic_dsff_channel, kan_input_dim, kernel_size=1),    # semantic_ch→256 for t3
            nn.Conv2d(self.semantic_dsff_channel, embed_dims[1], kernel_size=1)     # semantic_ch→320 for t4
        ])

        if self.use_detail_attention:
            self.detail_attention = DatasetAdaptiveAttention(
                channels=self.detail_dsff_channel,
                dataset_type=self.dataset_type,
                attention_alpha=attention_alpha
            )
            print(f"✅ : {self.detail_dsff_channel}, : {self.dataset_type}")



        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        
        if self.deep_supervision:
            self.dsv1 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
            self.dsv2 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
            self.dsv3 = nn.Conv2d(embed_dims[0]//4, num_classes, kernel_size=1)
            self.dsv4 = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)

    def forward(self, x):
        """

        :
            x: ， [B, C, H, W]

        :
            : ，
            : 
        """
        B = x.shape[0]
        input_size = x.size()[2:]
        deep_outputs = []

        if self.enable_wavelet:
            x = self.awep_preprocess(x)

        ### Encoder Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out

        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out

        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        detail_features = []
        semantic_features = []

        detail_features.append(self.detail_translayers[0](t1))
        detail_features.append(self.detail_translayers[1](t2))
        semantic_features.append(self.semantic_translayers[0](t3))

        ### Tokenized KAN Stage
        ### Stage 4
        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        semantic_features.append(self.semantic_translayers[1](t4))

        ### Bottleneck
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder Stage with DSFF Skip Connections

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode='bilinear'))

        semantic_dsff_out = self.semantic_dsff(semantic_features, semantic_features[1])
        semantic_expand = self.semantic_expand[1](semantic_dsff_out)
        out = torch.add(out, t4 + semantic_expand)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        dsv1 = out

        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2), mode='bilinear'))

        semantic_dsff_out = self.semantic_dsff(semantic_features, semantic_features[0])
        semantic_expand = self.semantic_expand[0](semantic_dsff_out)
        out = torch.add(out, t3 + semantic_expand)
        
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        dsv2 = out

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2,2), mode='bilinear'))

        detail_dsff_out = self.detail_dsff(detail_features, detail_features[1])
        if self.use_detail_attention and hasattr(self, 'detail_attention'):
            detail_dsff_out = self.detail_attention(detail_dsff_out)
        detail_expand = self.detail_expand[1](detail_dsff_out)
        out = torch.add(out, t2 + detail_expand)
        dsv3 = out

        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2,2), mode='bilinear'))

        detail_dsff_out = self.detail_dsff(detail_features, detail_features[0])
        if self.use_detail_attention and hasattr(self, 'detail_attention'):
            detail_dsff_out = self.detail_attention(detail_dsff_out)
        detail_expand = self.detail_expand[0](detail_dsff_out)
        out = torch.add(out, t1 + detail_expand)
        dsv4 = out
        
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2,2), mode='bilinear'))

        final_out = self.final(out)
        
        if self.deep_supervision:
            dsv1_out = F.interpolate(self.dsv1(dsv1), size=input_size, mode='bilinear', align_corners=False)
            dsv2_out = F.interpolate(self.dsv2(dsv2), size=input_size, mode='bilinear', align_corners=False)
            dsv3_out = F.interpolate(self.dsv3(dsv3), size=input_size, mode='bilinear', align_corners=False)
            dsv4_out = F.interpolate(self.dsv4(dsv4), size=input_size, mode='bilinear', align_corners=False)
            final_out = F.interpolate(final_out, size=input_size, mode='bilinear', align_corners=False)
            
            return [final_out, dsv4_out, dsv3_out, dsv2_out, dsv1_out]
        else:
            return final_out


# Backward-compatible aliases
UKAN_ASFF = UKDW
ASFF = DSFF
WaveletPreprocess = AWEP
WaveletDenoising = AWEPDenoising
WaveletTransform = AWEPWaveletTransform
