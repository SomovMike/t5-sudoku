from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config


class T5SudokuModel:
    def __init__(self, model_name="google-t5/t5-small", disable_dropout=True):
        """
        Initialize T5 model for Sudoku tasks with custom token mapping.
        
        Args:
            model_name (str): The name of the T5 model to load. Defaults to "google-t5/t5-small".
            disable_dropout (bool): Whether to disable dropout.
        """
        # Load the base model and tokenizer
        if disable_dropout:
            # Load config and disable all dropout
            config = T5Config.from_pretrained(model_name)
            config.dropout_rate = 0.0  # Disable main dropout
            config.layer_norm_epsilon = 1e-6  # Keep layer norm stable
            
            # Load model with no-dropout config
            self.model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.token_to_id_mapping = {
            '1': 2,
            '2': 3,
            '3': 4,
            '4': 5,
            'empty': 6
        }

        self.id_to_token_mapping = {v: k for k, v in self.token_to_id_mapping.items()}
        
        # Map new token IDs (2-6) to existing sudoku token embeddings
        self._setup_custom_embeddings()
        
        # Resize token embeddings to accommodate new tokens
        self.model.resize_token_embeddings(7)
    
    def _setup_custom_embeddings(self):
        """
        Set up custom embeddings for sudoku tokens by copying existing token embeddings
        to new token IDs (2-6).
        """
        for token, id in self.token_to_id_mapping.items():
            self.model.shared.weight.data[id] = self.model.shared.weight.data[self.tokenizer.encode(token)[0]]
            self.model.lm_head.weight.data[id] = self.model.lm_head.weight.data[self.tokenizer.encode(token)[0]]

        
