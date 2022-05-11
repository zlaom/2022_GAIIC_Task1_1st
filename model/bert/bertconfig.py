from torch import dropout


class BertConfig():
    '''
    Examples:
    >>> from transformers import BertModel, BertConfig
    >>> configuration = BertConfig()
    >>> model = BertModel(configuration)
    '''
    def __init__(
        self,
        vocab_size=460,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        output_attentions = False,
        output_hidden_states = False,
        image_dropout = 0.0
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.layer_norm_eps = layer_norm_eps
        # self.position_embedding_type = position_embedding_type
        # self.use_cache = use_cache
        # self.classifier_dropout = classifier_dropout
        self.output_attentions = output_attentions # 这两个output是我加的
        self.output_hidden_states = output_hidden_states
        self.image_dropout = image_dropout