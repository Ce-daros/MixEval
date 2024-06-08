from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model

@register_model("gpt1")
class GPT_1(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = "openai-community/openai-gpt"
        self.attn_implementation = None # If use default, set to None

        self.model = self.build_model().float32()
        self.model_max_len = self.model.config.n_ctx
        self.tokenizer = self.build_tokenizer()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length_closeend = min(
            self.model_max_len,
            self.max_input_length
        ) - self.closeended_max_new_tokens

