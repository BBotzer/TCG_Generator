import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Dataset Example
# ----------------------------
class MagicOracleDataset(Dataset):
    """
    A simple dataset for Magic: The Gathering oracle texts.
    Each item in the dataset is assumed to be a list (or tensor) of token IDs.
    """
    def __init__(self, tokenized_texts, max_seq_len):
        """
        tokenized_texts: list of lists or tensors of token ids
        max_seq_len: maximum sequence length (for padding or truncation)
        """
        self.tokenized_texts = tokenized_texts
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        # Get token ids and pad/truncate to max_seq_len
        tokens = self.tokenized_texts[idx]
        tokens = tokens[:self.max_seq_len]
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(tokens)
        # Pad if needed
        if len(tokens) < self.max_seq_len:
            pad_length = self.max_seq_len - len(tokens)
            tokens = tokens + [0] * pad_length  # assuming 0 is the pad token id
            attention_mask = attention_mask + [0] * pad_length
        tokens = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        return tokens, attention_mask

# ----------------------------
# Positional Encoding Module
# ----------------------------
class PositionalEncoding(nn.Module):
    """
    This module injects some information about the relative or absolute position
    of the tokens in the sequence. The positional encodings have the same dimension
    as the embeddings, so that the two can be summed.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# ----------------------------
# Decoder-only Transformer Model (GPT-style)
# ----------------------------
class GPTTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        # Create transformer decoder layers.
        # Note: We use nn.TransformerDecoderLayer and stack them.
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)
        # LM head (project back to vocabulary)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len) (optional)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()

        # Create embeddings and add positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)  # (batch_size, seq_len, d_model)

        # Transformer in PyTorch expects inputs as (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)

        # Generate causal mask to prevent positions from attending to future tokens
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)  # (seq_len, seq_len)

        # Optionally, incorporate an attention mask for padded tokens (if provided)
        # PyTorch's transformer decoder takes an "memory_key_padding_mask" for the encoder,
        # but here we are self-attending. Instead, we pass the mask in the "tgt_key_padding_mask".
        if attention_mask is not None:
            # Invert mask: True for positions that should be masked.
            tgt_key_padding_mask = ~attention_mask  # shape: (batch_size, seq_len)
        else:
            tgt_key_padding_mask = None

        # Note: Since we are using a decoder-only architecture without encoder memory,
        # we pass None for the memory argument.
        x = self.transformer_decoder(
            tgt=x,
            memory=None,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        x = x.transpose(0, 1)  # back to (batch_size, seq_len, d_model)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text autoregressively given an initial sequence.
        Args:
            input_ids: Tensor of shape (1, seq_len) as the prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: If provided, perform top-k sampling.
        Returns:
            output_ids: Tensor containing the original prompt plus generated tokens.
        """
        self.eval()
        generated = input_ids
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Crop the sequence to the model's max_seq_len if needed
            current_input = generated[:, -self.max_seq_len:]
            logits = self.forward(current_input)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (1, vocab_size) take the logits for the last token

            # Optionally adjust logits by temperature
            logits = logits / temperature

            # Optionally do top-k filtering
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter_(1, indices, F.softmax(values, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)  # sample next token
            generated = torch.cat((generated, next_token), dim=1)

            # Optionally, you could break if the generated token is the EOS token.
            # if next_token.item() == EOS_TOKEN_ID:
            #     break

        return generated

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == '__main__':
    # Hyperparameters
    VOCAB_SIZE = 10000  # change this to your vocabulary size
    D_MODEL = 512
    NHEAD = 8
    NUM_LAYERS = 6
    DROPOUT = 0.1
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 16
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-4

    # Assume you have preprocessed your data into a list of tokenized texts.
    # For demonstration, we'll create dummy data.
    dummy_tokenized_texts = [
        [1, 23, 456, 789, 12, 34, 2],
        [1, 67, 89, 45, 23, 2],
        # Add as many sequences as needed.
    ]
    dataset = MagicOracleDataset(dummy_tokenized_texts, max_seq_len=MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPTTransformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                             num_layers=NUM_LAYERS, dropout=DROPOUT, max_seq_len=MAX_SEQ_LEN)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop (simple language modeling objective: next token prediction)
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            input_ids, attention_mask = batch  # input_ids: (B, seq_len)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Shift input to create targets. For example, if input is: [t0, t1, ..., t_{n-1}],
            # then target is: [t1, t2, ..., t_n]. We ignore the last token.
            targets = input_ids[:, 1:].contiguous()
            model_inputs = input_ids[:, :-1].contiguous()
            # Also crop the attention mask accordingly
            tgt_attention_mask = attention_mask[:, :-1].contiguous()

            optimizer.zero_grad()
            logits = model(model_inputs, attention_mask=tgt_attention_mask)  # (B, seq_len-1, vocab_size)
            # Flatten the logits and targets for computing cross entropy
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=0)  # assuming pad token id is 0
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    # ----------------------------
    # Text Generation Example
    # ----------------------------
    # Suppose you want to generate text given a prompt.
    # Here, we create a dummy prompt. Replace with actual token IDs.
    prompt = torch.tensor([[1, 23, 456]], dtype=torch.long).to(device)  # shape (1, seq_len)
    generated_sequence = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=50)
    print("Generated token IDs:", generated_sequence)
