import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPPConv(nn.Module):
    """Single branch of the Atrous Spatial Pyramid Pooling module — a dilated 3x3 convolution."""
    
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global Average Pooling branch of ASPP — captures image-level context."""
    
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        # No BatchNorm here — after global avg pool, spatial dims are 1×1
        # which is incompatible with BN during training. Normalization happens
        # later in the ASPP projection layer after concatenation.
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLabV3+ style).
    
    Applies parallel dilated convolutions at multiple rates to capture
    multi-scale context, which is critical for detecting objects of
    varying sizes (e.g. small screws vs. large pliers) in the same scene.
    
    Branches:
        1. 1×1 conv         (rate=1)  — fine/local features
        2. 3×3 dilated conv  (rate=6)  — medium-range context
        3. 3×3 dilated conv  (rate=12) — large-range context
        4. 3×3 dilated conv  (rate=18) — very large-range context
        5. Global avg pooling           — image-level context
        
    All branches are concatenated and fused via a 1×1 projection.
    """
    
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super(ASPP, self).__init__()
        
        modules = []
        
        # Branch 1: 1×1 convolution (no dilation)
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))
        
        # Branches 2-4: Dilated 3×3 convolutions at different rates
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, dilation=rate))
        
        # Branch 5: Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.branches = nn.ModuleList(modules)
        
        # Projection: fuse all 5 branches → out_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Regularization — important for small datasets
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outputs, dim=1)
        return self.project(x)


class GraspHeatmapNet(nn.Module):
    """
    ResNet-50 Encoder + DeepLabV3+ Decoder with ASPP for dense grasp heatmap prediction.
    
    Architecture:
        Encoder: ResNet-50 backbone (pretrained ImageNet) with multi-scale feature extraction
        ASPP:    Atrous Spatial Pyramid Pooling on layer4 output for multi-scale context
        Decoder: DeepLabV3+ style — fuses ASPP output with low-level features from layer1,
                 then progressively upsamples to full resolution
                 
    Feature Pyramid (from 320×320 input):
        layer1: 80×80  × 256   ← low-level features (edges, textures)
        layer2: 40×40  × 512
        layer3: 20×20  × 1024
        layer4: 10×10  × 2048  ← high-level semantic features → ASPP
        
    Output channels (320 × 320):
        Ch 0: Graspability heatmap  (Sigmoid → [0, 1])
        Ch 1: sin(2θ)              (Tanh → [-1, 1])
        Ch 2: cos(2θ)              (Tanh → [-1, 1])
        Ch 3: Normalized width     (Sigmoid → [0, 1])
    """
    
    def __init__(self, pretrained=True):
        super(GraspHeatmapNet, self).__init__()
        
        # === ENCODER: ResNet-50 backbone with multi-scale feature taps ===
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        
        # Split encoder into stages so we can tap intermediate features
        # Stage 0: conv1 + bn1 + relu + maxpool → (B, 64, 80, 80)
        self.encoder_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        
        # Individual ResNet layers for multi-scale feature extraction
        self.encoder_layer1 = resnet.layer1  # (B, 256, 80, 80)  — low-level
        self.encoder_layer2 = resnet.layer2  # (B, 512, 40, 40)
        self.encoder_layer3 = resnet.layer3  # (B, 1024, 20, 20)
        self.encoder_layer4 = resnet.layer4  # (B, 2048, 10, 10) — high-level

        # Provide a unified `encoder` attribute so train.py's freezing logic works
        # This Sequential mirrors the old encoder for parameter iteration
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        
        # === ASPP: Multi-scale context aggregation on deepest features ===
        self.aspp = ASPP(in_channels=2048, out_channels=256, rates=(6, 12, 18))
        
        # === DECODER: DeepLabV3+ style ===
        
        # Low-level feature reduction (layer1: 256ch → 48ch)
        # Reduces channel count so low-level features don't overwhelm ASPP semantics
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        
        # Decoder refinement: fuses ASPP (256ch) + low-level (48ch) = 304ch → 256ch
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # === OUTPUT HEADS ===
        # Separate 1×1 convs for each output type with task-specific activations
        self.heatmap_head = nn.Conv2d(256, 1, kernel_size=1)  # Graspability
        self.angle_head   = nn.Conv2d(256, 2, kernel_size=1)  # sin(2θ), cos(2θ)
        self.width_head   = nn.Conv2d(256, 1, kernel_size=1)  # Normalized gripper width
        
    def forward(self, x):
        input_size = x.shape[2:]  # (320, 320)
        
        # === Encoder: extract multi-scale features ===
        x = self.encoder_stem(x)           # (B, 64, 80, 80)
        low_level = self.encoder_layer1(x) # (B, 256, 80, 80) — tap for decoder
        x = self.encoder_layer2(low_level) # (B, 512, 40, 40)
        x = self.encoder_layer3(x)         # (B, 1024, 20, 20)
        x = self.encoder_layer4(x)         # (B, 2048, 10, 10)
        
        # === ASPP: capture multi-scale context ===
        aspp_out = self.aspp(x)            # (B, 256, 10, 10)
        
        # === DeepLabV3+ Decoder ===
        # Step 1: Upsample ASPP output 8× to match low-level feature resolution
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[2:],
                                mode='bilinear', align_corners=False)  # (B, 256, 80, 80)
        
        # Step 2: Reduce low-level features
        low_level_reduced = self.low_level_conv(low_level)  # (B, 48, 80, 80)
        
        # Step 3: Concatenate and refine
        fused = torch.cat([aspp_up, low_level_reduced], dim=1)  # (B, 304, 80, 80)
        decoded = self.decoder(fused)  # (B, 256, 80, 80)
        
        # Step 4: Upsample to full input resolution
        decoded = F.interpolate(decoded, size=input_size,
                                mode='bilinear', align_corners=False)  # (B, 256, 320, 320)
        
        # === Output heads with task-specific activations ===
        heatmap = torch.sigmoid(self.heatmap_head(decoded))  # (B, 1, 320, 320) → [0, 1]
        angle   = torch.tanh(self.angle_head(decoded))       # (B, 2, 320, 320) → [-1, 1]
        width   = torch.sigmoid(self.width_head(decoded))    # (B, 1, 320, 320) → [0, 1]
        
        # Concatenate: [heatmap, sin2θ, cos2θ, width]
        out = torch.cat([heatmap, angle, width], dim=1)      # (B, 4, 320, 320)
        return out


if __name__ == "__main__":
    # Quick sanity check — verify shapes and parameter counts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraspHeatmapNet(pretrained=False).to(device)
    
    dummy = torch.randn(2, 3, 320, 320).to(device)
    with torch.no_grad():
        out = model(dummy)
    
    print(f"Input shape:  {dummy.shape}")     # (2, 3, 320, 320)
    print(f"Output shape: {out.shape}")       # (2, 4, 320, 320)
    print(f"Heatmap range: [{out[:, 0].min():.3f}, {out[:, 0].max():.3f}]")
    print(f"sin2t range:   [{out[:, 1].min():.3f}, {out[:, 1].max():.3f}]")
    print(f"cos2t range:   [{out[:, 2].min():.3f}, {out[:, 2].max():.3f}]")
    print(f"Width range:   [{out[:, 3].min():.3f}, {out[:, 3].max():.3f}]")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ASPP branch count
    print(f"ASPP branches:        {len(model.aspp.branches)}")
