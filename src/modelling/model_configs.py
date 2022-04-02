# Stores the model configs
class GeneralConfig:
    def __init__(self, **kwargs):
        self.num_classes = kwargs.pop("num_classes", None)
        assert self.num_classes, "num_classes must not be None!"
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.hidden_dropout_prob = kwargs.pop("hidden_dropout_prob", 0.1)
        self.layer_norm_eps = kwargs.pop("layer_norm_eps", 1e-12)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)


class StltModelConfig(GeneralConfig):
    def __init__(self, **kwargs):
        super(StltModelConfig, self).__init__(**kwargs)
        self.unique_categories = kwargs.pop("unique_categories", None)
        assert self.unique_categories, "unique_categories must not be None!"
        self.num_spatial_layers = kwargs.pop("num_spatial_layers", 4)
        self.num_temporal_layers = kwargs.pop("num_temporal_layers", 8)
        self.layout_num_frames = kwargs.pop("layout_num_frames", 256)
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
            f"- Max number of layout frames: {self.layout_num_frames}\n"
            f"- The backbone path is: {self.load_backbone_path}\n"
            f"- Freezing the backbone: {self.freeze_backbone}"
        )


class AppearanceModelConfig(GeneralConfig):
    def __init__(self, **kwargs):
        super(AppearanceModelConfig, self).__init__(**kwargs)
        self.appearance_num_frames = kwargs.pop("appearance_num_frames", None)
        assert self.appearance_num_frames, "appearance_num_frames must not be None!"
        self.resnet_model_path = kwargs.pop("resnet_model_path", None)
        assert self.resnet_model_path, "resnet_model_path must be provided"
        self.num_appearance_layers = kwargs.pop("num_appearance_layers", 4)

    def __repr__(self):
        return (
            f"- Number of classes: {self.num_classes}\n"
            f"- Max number of appearance frames: {self.appearance_num_frames}\n"
            f"- Hidden size: {self.hidden_size}\n"
            f"- If Transformer: Number of attention heads: {self.num_attention_heads}\n"
            f"- If Transformer: Number of layers: {self.num_appearance_layers}\n"
            f"- If Transformer: Hidden dropout probability: {self.hidden_dropout_prob}\n"
            f"- If Transformer: Layer norm eps: {self.layer_norm_eps}"
        )


class MultimodalModelConfig(GeneralConfig):
    def __init__(self, **kwargs):
        super(MultimodalModelConfig, self).__init__(**kwargs)
        # Not perfect way of creating the configs...
        self.stlt_config = StltModelConfig(**kwargs)
        self.appearance_config = AppearanceModelConfig(**kwargs)
        self.num_fusion_layers = kwargs.pop("num_fusion_layers", 4)
        self.load_backbone_path = kwargs.pop("load_backbone_path", None)
        self.freeze_backbone = kwargs.pop("freeze_backbone", False)

    def __repr__(self):
        stlt_repr = "*** Layout branch config: \n" + self.stlt_config.__repr__() + "\n"
        appearance_repr = (
            "*** Appearance branch config: \n"
            + self.appearance_config.__repr__()
            + "\n"
        )

        return (
            stlt_repr
            + appearance_repr
            + "*** Fusion module config: \n"
            + f"- Num fusion layers: {self.num_fusion_layers}\n"
            + f"- The backbone path is: {self.load_backbone_path}\n"
            + f"- Freezing the backbone: {self.freeze_backbone}"
        )


model_configs_factory = {
    "stlt": StltModelConfig,
    "resnet3d": AppearanceModelConfig,
    "resnet3d-transformer": AppearanceModelConfig,
    "lcf": MultimodalModelConfig,
    "caf": MultimodalModelConfig,
    "cacnf": MultimodalModelConfig,
}
