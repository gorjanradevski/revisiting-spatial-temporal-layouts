class StltModelConfig:
    def __init__(self, **kwargs):
        self.unique_categories = kwargs.pop("unique_categories", None)
        self.num_classes = kwargs.pop("num_classes", None)
        assert (
            self.unique_categories and self.num_classes
        ), "num_classes and unique_categories must not be None!"
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


class AppearanceModelConfig:
    def __init__(self, **kwargs):
        self.num_classes = kwargs.pop("num_classes", None)
        self.num_frames = kwargs.pop("num_frames", None)
        assert (
            self.num_classes and self.num_frames
        ), "num_classes and num_frames must not be None!"
        self.resnet_model_path = kwargs.pop(
            "resnet_model_path", "models/r3d50_KMS_200ep.pth"
        )
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)
        self.num_appearance_layers = kwargs.pop("num_appearance_layers", 2)
        self.hidden_dropout_prob = kwargs.pop("hidden_dropout_prob", 0.1)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)

    def __repr__(self):
        return (
            f"- Number of classes: {self.num_classes}\n"
            f"- Number of frames: {self.num_frames}\n"
            f"- Hidden size: {self.hidden_size}\n"
            f"- If Transformer: Number of attention heads: {self.num_attention_heads}\n"
            f"- If Transformer: Number of layers: {self.num_appearance_layers}\n"
            f"- If Transformer: Hidden dropout probability: {self.hidden_dropout_prob}\n"
            f"- If Transformer: Layer norm eps: {self.layer_norm_eps}"
        )


model_configs_factory = {"appearance": AppearanceModelConfig, "layout": StltModelConfig}
