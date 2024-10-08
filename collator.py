import logging
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import argparse

logger = logging.getLogger(__name__)

@dataclass
class EncodeCollator:
    data_args: str
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        text_ids = [x[0] for x in features]

        if self.data_args.lower_text:
            texts = [x[1].lower() for x in features]
        else:
            texts = [x[1] for x in features]

        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        
        collated_texts = self.tokenizer(
            texts,
            padding=False, 
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True, 
            # pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return text_ids, collated_texts