from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from utils.model_utils import generate_square_subsequent_mask

from modelling.configs import (
    AppearanceModelConfig,
    MultimodalModelConfig,
    StltModelConfig,
)
from modelling.resnets3d import generate_model


class CategoryBoxEmbeddings(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(CategoryBoxEmbeddings, self).__init__()
        self.category_embeddings = nn.Embedding(
            embedding_dim=config.hidden_size,
            num_embeddings=config.unique_categories,
            padding_idx=0,
        )
        self.box_embedding = nn.Linear(4, config.hidden_size)
        self.score_embeddings = nn.Linear(1, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        category_embeddings = self.category_embeddings(batch["categories"])
        boxes_embeddings = self.box_embedding(batch["boxes"])
        embeddings = category_embeddings + boxes_embeddings
        if "scores" in batch:
            score_embeddings = self.score_embeddings(batch["scores"].unsqueeze(-1))
            embeddings += score_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class SpatialTransformer(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(SpatialTransformer, self).__init__()
        self.category_box_embeddings = CategoryBoxEmbeddings(config)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=config.num_spatial_layers
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Num. boxes, Hidden size]
        cb_embeddings = self.category_box_embeddings(batch)
        num_frames, num_boxes, hidden_size = cb_embeddings.size()[1:]
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        cb_embeddings = cb_embeddings.flatten(0, 1)
        # [Num. boxes, Batch size x Num. frames, Hidden size]
        cb_embeddings = cb_embeddings.transpose(0, 1)
        # [Batch size x Num. frames, Num. boxes]
        src_key_padding_mask_boxes = batch["src_key_padding_mask_boxes"].flatten(0, 1)
        # [Num. boxes, Batch size x Num. frames, Hidden size]
        layout_embeddings = self.transformer(
            src=cb_embeddings,
            src_key_padding_mask=src_key_padding_mask_boxes,
        )
        # [Batch size x Num. frames, Num. boxes, Hidden size]
        layout_embeddings = layout_embeddings.transpose(0, 1)
        # [Batch size, Num. frames, Num. boxes, Hidden_size]
        layout_embeddings = layout_embeddings.view(
            -1, num_frames, num_boxes, hidden_size
        )
        # [Batch size, Num. frames, Hidden size]
        layout_embeddings = layout_embeddings[:, :, 0, :]

        return layout_embeddings


class FramesEmbeddings(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(FramesEmbeddings, self).__init__()
        self.layout_embedding = SpatialTransformer(config)
        self.position_embeddings = nn.Embedding(
            config.layout_num_frames, config.hidden_size
        )
        self.frame_type_embedding = nn.Embedding(5, config.hidden_size, padding_idx=0)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.layout_num_frames).expand((1, -1))
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Hidden size]
        layouts_embeddings = self.layout_embedding(batch)
        # Frame type and position embeddings
        frame_types_embeddings = self.frame_type_embedding(batch["frame_types"])
        num_frames = frame_types_embeddings.size()[1]
        position_embeddings = self.position_embeddings(
            self.position_ids[:, :num_frames]
        )
        # Preparing everything together
        embeddings = layouts_embeddings + position_embeddings + frame_types_embeddings
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings


class StltBackbone(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(StltBackbone, self).__init__()
        self.frames_embeddings = FramesEmbeddings(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
        )
        # Temporal Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.num_temporal_layers
        )

    @classmethod
    def from_pretrained(cls, config: StltModelConfig):
        model = cls(config)
        model.load_state_dict(torch.load(config.load_backbone_path, map_location="cpu"))
        return model

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Batch size, Num. frames, Hidden size]
        frames_embeddings = self.frames_embeddings(batch)
        # [Num. frames, Batch size, Hidden size]
        frames_embeddings = frames_embeddings.transpose(0, 1)
        # [Num. frames, Num. frames]
        causal_mask_frames = generate_square_subsequent_mask(
            frames_embeddings.size()[0]
        ).to(frames_embeddings.device)
        # [Num. frames, Batch size, Hidden size]
        transformer_output = self.transformer(
            src=frames_embeddings,
            mask=causal_mask_frames,
            src_key_padding_mask=batch["src_key_padding_mask_frames"],
        )

        return transformer_output


class ClassificationHead(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, hidden_state: torch.Tensor):
        return self.fc2(self.layer_norm(F.gelu(self.fc1(hidden_state))))


class Stlt(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(Stlt, self).__init__()
        self.config = config
        if config.load_backbone_path is not None:
            self.backbone = StltBackbone.from_pretrained(config)
            if config.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        else:
            self.backbone = StltBackbone(config)
        self.prediction_head = ClassificationHead(config)
        self.logit_names = ("stlt",)

    def train(self, mode: bool):
        super(Stlt, self).train(mode)
        if self.config.load_backbone_path and self.config.freeze_backbone:
            self.backbone.train(False)

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Num. frames, Batch size, Hidden size]
        stlt_output = self.backbone(batch)
        # [Batch size, Hidden size]
        batches = torch.arange(batch["categories"].size()[0]).to(
            batch["categories"].device
        )
        stlt_output = stlt_output[batch["lengths"] - 1, batches, :]
        logits = (self.prediction_head(stlt_output),)

        return {k: v for k, v in zip(self.logit_names, logits)}


class Resnet3D(nn.Module):
    def __init__(self, config: AppearanceModelConfig):
        super(Resnet3D, self).__init__()
        resnet = generate_model(model_depth=50, n_classes=1139)
        resnet.load_state_dict(
            torch.load(config.resnet_model_path, map_location="cpu")["state_dict"]
        )
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        for module in self.resnet.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
        if config.num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.classifier = nn.Linear(2048, config.num_classes)
        self.logit_names = ("resnet3d",)

    def train(self, mode: bool):
        super(Resnet3D, self).train(mode)
        for module in self.resnet.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.train(False)

    def forward_features(self, batch):
        return self.resnet(batch["video_frames"])

    def forward(self, batch):
        features = self.forward_features(batch)
        features = self.avgpool(features).flatten(1)
        logits = (self.classifier(features),)

        return {k: v for k, v in zip(self.logit_names, logits)}


class TransformerResnet(nn.Module):
    def __init__(self, config: AppearanceModelConfig):
        super(TransformerResnet, self).__init__()
        self.resnet = Resnet3D(config)
        self.projector = nn.Conv3d(
            in_channels=2048, out_channels=config.hidden_size, kernel_size=(1, 1, 1)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config.num_appearance_layers
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(config.appearance_num_frames + 1, 1, config.hidden_size)
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def forward_features(self, batch):
        # We need the batch size for the CLS token
        B = batch["video_frames"].shape[0]
        features = self.resnet.forward_features(batch)
        # [Batch size, Hidden size, Temporal., Spatial., Spatial.]
        features = self.projector(features)
        # [Batch size, Hidden size, Seq. len]
        features = features.flatten(2)
        # [Seq. len, Batch size, Hidden size]
        features = features.permute(2, 0, 1)
        cls_tokens = self.cls_token.expand(
            -1, B, -1
        )  # stole cls_tokens impl from Ross Wightman thanks
        features = torch.cat((cls_tokens, features), dim=0)
        features = features + self.pos_embed
        # [Seq. len, Batch size, Hidden size]
        features = self.transformer(src=features)

        return features

    def forward(self, batch):
        # [Seq. len, Batch size, Hidden size]
        features = self.forward_features(batch)
        # [Batch size, Hidden size]
        cls_state = features[0, :, :]
        logits = (self.classifier(cls_state),)

        return {k: v for k, v in zip(self.resnet.logit_names, logits)}

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}


class FusionHead(nn.Module):
    def __init__(self, config: MultimodalModelConfig):
        super(FusionHead, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, hidden_state: torch.Tensor):
        return self.fc2(self.layer_norm(F.gelu(self.fc1(hidden_state))))


class LateConcatenationFusion(nn.Module):
    # LCF
    def __init__(self, config: MultimodalModelConfig):
        super(LateConcatenationFusion, self).__init__()
        self.layout_branch = StltBackbone(config.stlt_config)
        self.appearance_branch = TransformerResnet(config.appearance_config)
        self.classifier = FusionHead(config)
        self.logit_names = ("lcf",)

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Num. Lay. frames, Batch size, Hidden size]
        layout_output = self.layout_branch(batch)
        # [Batch size, Hidden size]
        batches = torch.arange(batch["categories"].size()[0]).to(
            batch["categories"].device
        )
        layout_output = layout_output[batch["lengths"] - 1, batches, :]
        # [Num. App. frames, Batch size, Hidden size]
        appearance_output = self.appearance_branch.forward_features(batch)
        # [Batch size, Hidden size]
        appearance_output = appearance_output[0, :, :]
        # [Batch size, Hidden size * 2]
        fused_features = torch.cat((layout_output, appearance_output), dim=-1)
        logits = (self.classifier(fused_features),)

        return {k: v for k, v in zip(self.logit_names, logits)}


# CAF and CACNF, and related modules


class FeedforwardModule(nn.Module):
    def __init__(self, config: MultimodalModelConfig):
        super(FeedforwardModule, self).__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.linear2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs: torch.Tensor):
        hidden_states = self.dropout(self.linear2(F.gelu(self.linear1(inputs))))
        hidden_states = self.ln(hidden_states + inputs)
        return hidden_states


class SelfAttentionLayer(nn.Module):
    def __init__(self, config: MultimodalModelConfig):
        super(SelfAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.hidden_dropout_prob,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs, causal_mask=None, key_padding_mask=None):
        hidden_states = self.attn(
            inputs,
            inputs,
            inputs,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
        )[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.ln(hidden_states + inputs)

        return hidden_states


class CrossAttentionLayer(nn.Module):
    def __init__(self, config: MultimodalModelConfig):
        super(CrossAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.hidden_dropout_prob,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs, context, context_padding_mask=None):
        hidden_states = self.attn(
            inputs,
            context,
            context,
            key_padding_mask=context_padding_mask,
        )[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.ln(hidden_states + inputs)

        return hidden_states


class CrossModalModule(nn.Module):
    def __init__(self, config: MultimodalModelConfig):
        super(CrossModalModule, self).__init__()
        # Cross-attention
        self.cross_attn = CrossAttentionLayer(config)
        # Layout-Appearance self-attention
        self.layout_attn = SelfAttentionLayer(config)
        self.layout_ffn = FeedforwardModule(config)
        # Appearance-Layout self-attention
        self.appearance_attn = SelfAttentionLayer(config)
        self.appearance_ffn = SelfAttentionLayer(config)

    def forward(
        self,
        layout_hidden_states,
        appearance_hidden_states,
        causal_attn_mask_layout,
        src_key_padding_mask_layout,
    ):
        # Cross-attention
        layout_attn_output = self.cross_attn(
            layout_hidden_states,
            appearance_hidden_states,
        )
        appearance_attn_output = self.cross_attn(
            appearance_hidden_states,
            layout_hidden_states,
            src_key_padding_mask_layout,
        )
        # Self-attention
        layout_attn_output = self.layout_attn(
            layout_attn_output,
            causal_mask=causal_attn_mask_layout,
            key_padding_mask=src_key_padding_mask_layout,
        )
        appearance_attn_output = self.appearance_attn(appearance_attn_output)
        # Feed-forward
        layout_output = self.layout_ffn(layout_attn_output)
        appearance_output = self.appearance_ffn(appearance_attn_output)

        return layout_output, appearance_output


class CrossAttentionFusionBackbone(nn.Module):
    # Backbone for CAF and CACNF
    def __init__(self, config: MultimodalModelConfig):
        super(CrossAttentionFusionBackbone, self).__init__()
        # Unimodal embeddings
        self.layout_branch = StltBackbone(config.stlt_config)
        self.appearance_branch = TransformerResnet(config.appearance_config)
        # Multimodal embeddings
        self.mm_fusion = nn.ModuleList(
            [CrossModalModule(config) for _ in range(config.num_fusion_layers)]
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        causal_mask_frames = generate_square_subsequent_mask(
            batch["categories"].size()[1]
        ).to(batch["categories"].device)
        # [Lay. num. frames, Batch size, Hidden size]
        layout_hidden_states = self.layout_branch(batch)
        # [App. num. frames, Batch size, Hidden size]
        appearance_hidden_states = self.appearance_branch.forward_features(batch)
        # Get hidden states for individual branches
        # [Batch size, Hidden size]
        batches = torch.arange(batch["categories"].size()[0]).to(
            batch["categories"].device
        )
        layout_hidden_state = layout_hidden_states[batch["lengths"] - 1, batches, :]
        appearance_hidden_state = appearance_hidden_states[0, :, :]
        # Multimodal fusion
        for layer in self.mm_fusion:
            layout_hidden_states, appearance_hidden_states = layer(
                layout_hidden_states,
                appearance_hidden_states,
                causal_mask_frames,
                batch["src_key_padding_mask_frames"],
            )
        # Get fused hidden state
        # [Batch size, Hidden size]
        last_fused_state = torch.cat(
            (
                layout_hidden_states[batch["lengths"] - 1, batches, :],
                appearance_hidden_states[0, :, :],
            ),
            dim=-1,
        )

        return {
            "layout_hidden_state": layout_hidden_state,
            "appearance_hidden_state": appearance_hidden_state,
            "last_fused_state": last_fused_state,
        }


class CrossAttentionFusion(nn.Module):
    # CAF
    def __init__(self, config: MultimodalModelConfig):
        super(CrossAttentionFusion, self).__init__()
        self.caf_backbone = CrossAttentionFusionBackbone(config)
        # Classifier
        self.classifier = FusionHead(config)
        self.logit_names = ("caf",)

    def forward(self, batch: Dict[str, torch.Tensor]):
        cross_attention_fusion_embeddings = self.caf_backbone(batch)
        # [Batch size, Hidden size * 2]
        last_fused_state = cross_attention_fusion_embeddings["last_fused_state"]
        logits = (self.classifier(last_fused_state),)

        return {k: v for k, v in zip(self.logit_names, logits)}


class CrossAttentionCentralNetFusion(nn.Module):
    # CACNF
    def __init__(self, config: MultimodalModelConfig):
        super(CrossAttentionCentralNetFusion, self).__init__()
        self.config = config
        if config.load_backbone_path is not None:
            self.backbone = CrossAttentionFusionBackbone.from_pretrained(config)
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.backbone = CrossAttentionFusionBackbone(config)
        # Classifiers
        self.layout_classifier = ClassificationHead(config)
        self.appearance_classifier = ClassificationHead(config)
        self.fusion_classifier = FusionHead(config)
        self.logit_names = ("stlt", "resnet3d", "caf", "ensemble")

    def train(self, mode: bool):
        super(CrossAttentionCentralNetFusion, self).train(mode)
        if self.config.load_backbone_path:
            self.backbone.train(False)

    def forward(self, batch: Dict[str, torch.Tensor]):
        cross_attention_fusion_embeddings = self.backbone(batch)
        logits = ()
        # Unimodal part
        # [Batch size, Num. classes]
        logits += (
            self.layout_classifier(
                cross_attention_fusion_embeddings["layout_hidden_state"]
            ),
        )
        logits += (
            self.appearance_classifier(
                cross_attention_fusion_embeddings["appearance_hidden_state"]
            ),
        )
        # Multimodal part
        # [Batch size, Hidden size * 2]
        last_fused_state = cross_attention_fusion_embeddings["last_fused_state"]
        # [Batch size, Num. classes]
        logits += (self.fusion_classifier(last_fused_state),)
        # Ensemble
        logits += (sum(logits) / 3,)

        return {k: v for k, v in zip(self.logit_names, logits)}


models_factory = {
    "stlt": Stlt,
    "resnet3d": Resnet3D,
    "resnet3d-transformer": TransformerResnet,
    "lcf": LateConcatenationFusion,
    "caf": CrossAttentionFusion,
    "cacnf": CrossAttentionCentralNetFusion,
}
