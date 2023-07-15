from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PreTrainedTokenizer, TensorType, is_torch_available
from transformers.configuration_utils import PretrainedConfig


class AlignableLlamaConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(self, das_layer, das_token_range, **kwargs):
        self.das_layer = das_layer

        super().__init__(**kwargs)
