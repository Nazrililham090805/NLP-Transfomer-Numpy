"""
Implementasi Transformer dari Nol dengan NumPy
"""

import numpy as np


class TokenEmbedding:
    """
    Token Embedding Layer
    Mengkonversi token IDs menjadi dense vectors
    """
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: ukuran vocabulary
            d_model: dimensi embedding
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Inisialisasi embedding matrix dengan distribusi normal
        self.embedding_matrix = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, token_ids):
        """
        Forward pass untuk token embedding

        Args:
            token_ids: array of token indices [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        return self.embedding_matrix[token_ids]


class PositionalEncoding:
    """
    Sinusoidal Positional Encoding
    Menambahkan informasi posisi ke embeddings
    """
    def __init__(self, max_seq_len, d_model):
        """
        Args:
            max_seq_len: panjang maksimum sequence
            d_model: dimensi embedding
        """
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.pos_encoding = self._create_positional_encoding()

    def _create_positional_encoding(self):
        """
        Membuat sinusoidal positional encoding matrix

        Returns:
            pos_encoding: [max_seq_len, d_model]
        """
        position = np.arange(self.max_seq_len)[:, np.newaxis]  # [max_seq_len, 1]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))

        pos_encoding = np.zeros((self.max_seq_len, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)  # even indices
        pos_encoding[:, 1::2] = np.cos(position * div_term)  # odd indices

        return pos_encoding

    def forward(self, embeddings):
        """
        Menambahkan positional encoding ke embeddings

        Args:
            embeddings: [batch_size, seq_len, d_model]

        Returns:
            embeddings + positional encoding: [batch_size, seq_len, d_model]
        """
        seq_len = embeddings.shape[1]
        return embeddings + self.pos_encoding[:seq_len, :]


def softmax(x, axis=-1):
    """
    Stable softmax implementation

    Args:
        x: input array
        axis: axis untuk softmax

    Returns:
        softmax probabilities
    """
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len):
    """
    Membuat causal mask untuk mencegah attention ke token masa depan

    Args:
        seq_len: panjang sequence

    Returns:
        mask: [seq_len, seq_len] dengan 0 untuk posisi yang diperbolehkan,
              -inf untuk posisi yang di-mask
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = mask * -1e9  # Gunakan nilai sangat negatif untuk mask
    return mask


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    def __init__(self, d_k):
        """
        Args:
            d_k: dimensi key (untuk scaling)
        """
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass untuk scaled dot-product attention

        Args:
            Q: Query [batch_size, seq_len, d_k]
            K: Key [batch_size, seq_len, d_k]
            V: Value [batch_size, seq_len, d_v]
            mask: optional mask [seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_v]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        # Hitung attention scores: Q @ K.T / sqrt(d_k)
        # Q: [batch, seq_len, d_k], K.T: [batch, d_k, seq_len]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_k)
        # scores: [batch_size, seq_len, seq_len]

        # Apply causal mask jika ada
        if mask is not None:
            scores = scores + mask  # Tambahkan -inf ke posisi yang di-mask

        # Apply softmax untuk mendapatkan attention weights
        attention_weights = softmax(scores, axis=-1)
        # attention_weights: [batch_size, seq_len, seq_len]

        # Hitung output: attention_weights @ V
        output = np.matmul(attention_weights, V)
        # output: [batch_size, seq_len, d_v]

        return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention
    Menjalankan attention secara parallel dengan beberapa "heads"
    """
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: dimensi model
            num_heads: jumlah attention heads
        """
        assert d_model % num_heads == 0, "d_model harus habis dibagi num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimensi per head

        # Weight matrices untuk Q, K, V projections
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01

        # Weight matrix untuk output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(self.d_k)

    def _split_heads(self, x):
        """
        Split menjadi multiple heads

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.shape
        # Reshape ke [batch_size, seq_len, num_heads, d_k]
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose ke [batch_size, num_heads, seq_len, d_k]
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        """
        Combine multiple heads kembali

        Args:
            x: [batch_size, num_heads, seq_len, d_k]

        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.shape
        # Transpose ke [batch_size, seq_len, num_heads, d_k]
        x = x.transpose(0, 2, 1, 3)
        # Reshape ke [batch_size, seq_len, d_model]
        return x.reshape(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        """
        Forward pass untuk multi-head attention

        Args:
            x: input [batch_size, seq_len, d_model]
            mask: optional causal mask [seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = x.shape[0]

        # Linear projections untuk Q, K, V
        Q = np.matmul(x, self.W_q)  # [batch_size, seq_len, d_model]
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)

        # Split into multiple heads
        Q = self._split_heads(Q)  # [batch_size, num_heads, seq_len, d_k]
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Apply attention untuk setiap head
        # Reshape untuk attention: [batch_size * num_heads, seq_len, d_k]
        Q_reshaped = Q.reshape(batch_size * self.num_heads, -1, self.d_k)
        K_reshaped = K.reshape(batch_size * self.num_heads, -1, self.d_k)
        V_reshaped = V.reshape(batch_size * self.num_heads, -1, self.d_k)

        # Compute attention
        attn_output, attn_weights = self.attention.forward(Q_reshaped, K_reshaped, V_reshaped, mask)
        # attn_output: [batch_size * num_heads, seq_len, d_k]
        # attn_weights: [batch_size * num_heads, seq_len, seq_len]

        # Reshape kembali
        seq_len = x.shape[1]
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len, self.d_k)
        attn_weights = attn_weights.reshape(batch_size, self.num_heads, seq_len, seq_len)

        # Combine heads
        concat_output = self._combine_heads(attn_output)  # [batch_size, seq_len, d_model]

        # Final linear projection
        output = np.matmul(concat_output, self.W_o)  # [batch_size, seq_len, d_model]

        return output, attn_weights


def gelu(x):
    """
    GELU (Gaussian Error Linear Unit) activation function
    GELU(x) = x * Φ(x), dimana Φ(x) adalah CDF dari distribusi normal standar

    Args:
        x: input array

    Returns:
        GELU activated output
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class FeedForwardNetwork:
    """
    Feed-Forward Network (FFN)
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    Dua lapisan linear dengan aktivasi non-linear di tengah
    """
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: dimensi model
            d_ff: dimensi hidden layer (biasanya 4 * d_model)
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Layer pertama: d_model -> d_ff
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros(d_ff)

        # Layer kedua: d_ff -> d_model
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """
        Forward pass untuk FFN

        Args:
            x: input [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # First linear layer + GELU activation
        hidden = gelu(np.matmul(x, self.W1) + self.b1)
        # hidden: [batch_size, seq_len, d_ff]

        # Second linear layer
        output = np.matmul(hidden, self.W2) + self.b2
        # output: [batch_size, seq_len, d_model]

        return output


class LayerNormalization:
    """
    Layer Normalization
    LayerNorm(x) = γ * (x - μ) / σ + β
    dimana μ dan σ dihitung per layer
    """
    def __init__(self, d_model, eps=1e-6):
        """
        Args:
            d_model: dimensi model
            eps: epsilon untuk numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)  # scale parameter
        self.beta = np.zeros(d_model)  # shift parameter

    def forward(self, x):
        """
        Forward pass untuk layer normalization

        Args:
            x: input [batch_size, seq_len, d_model]

        Returns:
            normalized output: [batch_size, seq_len, d_model]
        """
        # Hitung mean dan variance per layer (across d_model dimension)
        mean = np.mean(x, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]
        variance = np.var(x, axis=-1, keepdims=True)  # [batch_size, seq_len, 1]

        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.eps)

        # Scale and shift
        output = self.gamma * x_normalized + self.beta

        return output


class TransformerBlock:
    """
    Single Transformer Decoder Block dengan Pre-Normalization
    Architecture: LayerNorm -> MultiHeadAttention -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(self, d_model, num_heads, d_ff):
        """
        Args:
            d_model: dimensi model
            num_heads: jumlah attention heads
            d_ff: dimensi hidden layer FFN
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln1 = LayerNormalization(d_model)
        self.ln2 = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass untuk transformer block

        Args:
            x: input [batch_size, seq_len, d_model]
            mask: causal mask [seq_len, seq_len]

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Pre-norm + Multi-Head Attention + Residual
        ln_x = self.ln1.forward(x)
        attn_output, attn_weights = self.attention.forward(ln_x, mask)
        x = x + attn_output  # Residual connection

        # Pre-norm + FFN + Residual
        ln_x = self.ln2.forward(x)
        ffn_output = self.ffn.forward(ln_x)
        x = x + ffn_output  # Residual connection

        return x, attn_weights


class DecoderOnlyTransformer:
    """
    Decoder-Only Transformer (GPT-style)
    Complete implementation dari embedding hingga output
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        """
        Args:
            vocab_size: ukuran vocabulary
            d_model: dimensi model
            num_heads: jumlah attention heads
            d_ff: dimensi hidden layer FFN
            num_layers: jumlah transformer blocks
            max_seq_len: panjang maksimum sequence
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

        # Final layer normalization
        self.ln_final = LayerNormalization(d_model)

        # Output projection ke vocabulary size
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.01

    def forward(self, token_ids, return_attention=False):
        """
        Forward pass untuk decoder-only transformer

        Args:
            token_ids: input token IDs [batch_size, seq_len]
            return_attention: jika True, return juga attention weights

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            probs: [batch_size, vocab_size] - distribusi probabilitas token berikutnya
            attention_weights (optional): list of attention weights dari setiap layer
        """
        seq_len = token_ids.shape[1]

        # Create causal mask
        mask = create_causal_mask(seq_len)

        # Token embedding + Positional encoding
        x = self.token_embedding.forward(token_ids)
        x = self.pos_encoding.forward(x)
        # x: [batch_size, seq_len, d_model]

        # Pass through transformer blocks
        attention_weights_list = []
        for block in self.blocks:
            x, attn_weights = block.forward(x, mask)
            attention_weights_list.append(attn_weights)

        # Final layer normalization
        x = self.ln_final.forward(x)

        # Project to vocabulary size
        logits = np.matmul(x, self.output_projection)
        # logits: [batch_size, seq_len, vocab_size]

        # Get probability distribution untuk token berikutnya (posisi terakhir)
        last_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        probs = softmax(last_token_logits, axis=-1)  # [batch_size, vocab_size]

        if return_attention:
            return logits, probs, attention_weights_list
        else:
            return logits, probs
