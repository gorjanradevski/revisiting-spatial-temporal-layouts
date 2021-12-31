from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from utils.model_utils import generate_square_subsequent_mask


class StltModelConfig:
    def __init__(self, unique_categories: int, num_classes: int, **kwargs):
        self.num_classes = num_classes
        self.unique_categories = unique_categories
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.hidden_dropout_prob = kwargs.pop("hidden_dropout_prob", 0.1)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)
        self.num_spatial_layers = kwargs.pop("num_spatial_layers", 4)
        self.num_temporal_layers = kwargs.pop("num_temporal_layers", 8)
        self.max_num_frames = kwargs.pop("max_num_frames", 256)
        self.load_backbone_path = kwargs.pop("load_backbone_path", None)
        self.freeze_backbone = kwargs.pop("freeze_backbone", False)

    def __repr__(self):
        return (
            f"- Unique categories: {self.unique_categories}\n"
            f"- Number of classes: {self.num_classes}\n"
            f"- Hidden size: {self.hidden_size}\n"
            f"- Hidden dropout probability: {self.hidden_dropout_prob}\n"
            f"- Layer normalization epsilon: {self.layer_norm_eps}\n"
            f"- Number of attention heads: {self.num_attention_heads}\n"
            f"- Number of spatial layers: {self.num_spatial_layers}\n"
            f"- Number of temporal layers: {self.num_temporal_layers}\n"
            f"- Max number of frames accepted: {self.max_num_frames}\n"
            f"- The backbone path is: {self.load_backbone_path}\n"
            f"- Freezing the backbone: {self.freeze_backbone}"
        )


class CategoryBoxEmbeddings(nn.Module):
    def __init__(self, config: StltModelConfig):
        super(CategoryBoxEmbeddings, self).__init__()
        self.category_embeddings = nn.Embedding(
            embedding_dim=config.hidden_size,
            num_embeddings=config.unique_categories,
            padding_idx=0,
        )
        self.box_embedding = nn.Linear(4, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        category_embeddings = self.category_embeddings(batch["categories"])
        boxes_embeddings = self.box_embedding(batch["boxes"])
        embeddings = category_embeddings + boxes_embeddings
        if "video_features" in batch:
            embeddings += self.box_features_embedding(batch["video_features"])
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
            config.max_num_frames, config.hidden_size
        )
        self.frame_type_embedding = nn.Embedding(5, config.hidden_size, padding_idx=0)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_num_frames).expand((1, -1))
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
            self.stlt_backbone = StltBackbone.from_pretrained(config)
            if config.freeze_backbone:
                for param in self.stlt_output.parameters():
                    param.requires_grad = False
        else:
            self.stlt_backbone = StltBackbone(config)
        self.prediction_head = ClassificationHead(config)

    def train(self, mode: bool):
        super(Stlt, self).train(mode)
        if self.config.load_backbone_path and self.config.freeze_backbone:
            self.stlt_backbone.train(False)

    def forward(self, batch: Dict[str, torch.Tensor]):
        # [Num. frames, Batch size, Hidden size]
        stlt_output = self.stlt_backbone(batch)
        # [Batch size, Hidden size]
        batches = torch.arange(batch["categories"].size()[0]).to(
            batch["categories"].device
        )
        stlt_output = stlt_output[batch["lengths"] - 1, batches, :]

        return self.prediction_head(stlt_output)
