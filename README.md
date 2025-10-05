# Implementasi Transformer dari Nol dengan NumPy

Implementasi lengkap arsitektur Decoder-Only Transformer (GPT-style) dari nol menggunakan hanya NumPy, tanpa framework deep learning.

## Deskripsi

Proyek ini mengimplementasikan semua komponen utama dari arsitektur Transformer untuk tugas generasi teks (decoder-only), termasuk:

1. **Token Embedding** - Mengkonversi token IDs menjadi dense vectors
2. **Positional Encoding** - Sinusoidal encoding untuk informasi posisi
3. **Scaled Dot-Product Attention** - Mekanisme attention dengan scaling
4. **Multi-Head Attention** - Parallel attention dengan multiple heads
5. **Feed-Forward Network** - Dua lapisan linear dengan aktivasi GELU
6. **Layer Normalization** - Normalisasi per layer
7. **Residual Connections** - Skip connections untuk training stability
8. **Causal Masking** - Mencegah akses informasi dari token masa depan
9. **Output Layer** - Proyeksi ke vocabulary size dan distribusi softmax

## Struktur File

```
transformer_from_scratch/
├── transformer.py          # Implementasi lengkap semua komponen
├── test_transformer.py     # Testing dan validasi
└── README.md              # Dokumentasi ini
```

## Dependensi

Hanya membutuhkan NumPy:

```bash
pip install numpy
```

## Cara Menjalankan

### 1. Menjalankan Testing

Untuk memvalidasi semua komponen dan memverifikasi implementasi:

```bash
python test_transformer.py
```

Output akan menampilkan hasil testing untuk setiap komponen, termasuk:
- Validasi dimensi tensor
- Verifikasi causal masking
- Validasi softmax (sum = 1)
- Testing forward pass lengkap

### 2. Menggunakan Model

Contoh penggunaan model dalam kode Python:

```python
import numpy as np
from transformer import DecoderOnlyTransformer

# Inisialisasi model
vocab_size = 1000      # Ukuran vocabulary
d_model = 128          # Dimensi embedding
num_heads = 8          # Jumlah attention heads
d_ff = 512            # Dimensi FFN (biasanya 4 * d_model)
num_layers = 6        # Jumlah transformer blocks
max_seq_len = 100     # Panjang maksimum sequence

model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=max_seq_len
)

# Input: token IDs
batch_size = 2
seq_len = 10
token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

# Forward pass
logits, probs = model.forward(token_ids)

# Output:
# - logits: [batch_size, seq_len, vocab_size]
# - probs: [batch_size, vocab_size] distribusi probabilitas token berikutnya

# Prediksi token berikutnya
next_token = np.argmax(probs[0])  # Token dengan probabilitas tertinggi
print(f"Next token prediction: {next_token}")
```

### 3. Mengakses Attention Weights (Optional)

```python
# Forward pass dengan attention weights
logits, probs, attention_weights = model.forward(token_ids, return_attention=True)

# attention_weights: list of [batch_size, num_heads, seq_len, seq_len] untuk setiap layer
# Visualisasi attention pattern untuk layer pertama, head pertama
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0][0, 0], cmap='viridis')
plt.colorbar()
plt.title('Attention Pattern - Layer 1, Head 1')
plt.show()
```

## Arsitektur

### Decoder-Only Transformer (GPT-style)

Arsitektur ini menggunakan **pre-normalization** (layer norm sebelum sub-layer):

```
Input Token IDs
    ↓
Token Embedding + Positional Encoding
    ↓
┌─────────────────────────────────┐
│  Transformer Block (x N layers) │
│  ┌──────────────────────────┐   │
│  │ Layer Norm               │   │
│  │ Multi-Head Attention     │   │
│  │ Residual Connection      │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ Layer Norm               │   │
│  │ Feed-Forward Network     │   │
│  │ Residual Connection      │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
    ↓
Final Layer Norm
    ↓
Output Projection to Vocab Size
    ↓
Softmax → Probability Distribution
```

### Causal Masking

Model menggunakan causal mask untuk mencegah token melihat informasi dari token masa depan:

```
Position:  0  1  2  3  4
    0    [ 0  -  -  -  - ]
    1    [ 0  0  -  -  - ]
    2    [ 0  0  0  -  - ]
    3    [ 0  0  0  0  - ]
    4    [ 0  0  0  0  0 ]

0 = allowed, - = masked (-inf)
```

### Positional Encoding

Menggunakan **sinusoidal positional encoding** seperti pada paper "Attention is All You Need":

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Alasan pemilihan**: Sinusoidal encoding memungkinkan model untuk menggeneralisasi ke sequence yang lebih panjang dari yang dilihat saat training, karena menggunakan fungsi periodik yang dapat diekstrapolasi.

## Validasi dan Testing

File `test_transformer.py` melakukan validasi komprehensif:

### 1. Validasi Dimensi Tensor

Setiap komponen diverifikasi untuk memastikan output shape sesuai dengan expected:

```
Token Embedding:     [batch, seq_len] → [batch, seq_len, d_model]
Positional Encoding: [batch, seq_len, d_model] → [batch, seq_len, d_model]
Attention:          [batch, seq_len, d_model] → [batch, seq_len, d_model]
FFN:                [batch, seq_len, d_model] → [batch, seq_len, d_model]
Full Model:         [batch, seq_len] → [batch, seq_len, vocab_size]
```

### 2. Validasi Causal Mask

- Memverifikasi mask hanya memperbolehkan attention ke posisi ≤ current position
- Posisi masa depan di-mask dengan nilai -inf

### 3. Validasi Softmax

- Attention weights sum = 1.0 untuk setiap query position
- Output probability distribution sum = 1.0

### 4. Validasi Layer Normalization

- Mean ≈ 0
- Variance ≈ 1

## Hasil Testing

Semua komponen telah divalidasi dan passed:

```
[PASS] Token Embedding test passed!
[PASS] Positional Encoding test passed!
[PASS] Causal Mask test passed!
[PASS] Scaled Dot-Product Attention test passed!
[PASS] Multi-Head Attention test passed!
[PASS] Feed-Forward Network test passed!
[PASS] Layer Normalization test passed!
[PASS] Transformer Block test passed!
[PASS] Full Transformer test passed!
```

## Komponen Detail

### 1. Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

- Scaling dengan sqrt(d_k) untuk stability
- Causal mask ditambahkan sebelum softmax

### 2. Multi-Head Attention

- Membagi d_model menjadi num_heads
- Setiap head belajar representasi yang berbeda
- Hasil di-concatenate dan di-project

### 3. Feed-Forward Network

```
FFN(x) = GELU(xW1 + b1)W2 + b2
```

- Dua linear layers dengan GELU activation
- d_ff biasanya 4 × d_model

### 4. Layer Normalization

```
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β
```

- Normalisasi per layer (across d_model dimension)
- Learnable parameters γ (scale) dan β (shift)

## Konfigurasi Model

### Small Model (untuk testing):
- vocab_size: 100
- d_model: 64
- num_heads: 8
- d_ff: 256
- num_layers: 4

### Medium Model (GPT-2 Small style):
- vocab_size: 50257
- d_model: 768
- num_heads: 12
- d_ff: 3072
- num_layers: 12

### Large Model (GPT-2 Medium style):
- vocab_size: 50257
- d_model: 1024
- num_heads: 16
- d_ff: 4096
- num_layers: 24

## Limitasi

1. **Training**: Implementasi ini hanya forward pass, tidak termasuk backpropagation
2. **Performance**: NumPy lebih lambat dibanding framework optimized seperti PyTorch/TensorFlow
3. **GPU**: Tidak ada dukungan GPU acceleration
4. **Tokenization**: Perlu implementasi tokenizer terpisah untuk text input

## Pengembangan Lebih Lanjut (Bonus Features)

Beberapa fitur yang bisa ditambahkan:

1. **Visualisasi Attention Patterns** - Heatmap untuk melihat apa yang di-attend model
2. **Weight Tying** - Share weights antara token embedding dan output projection
3. **RoPE (Rotary Position Embedding)** - Alternatif positional encoding yang lebih baik
4. **Dropout** - Regularization untuk training
5. **Backpropagation** - Implementasi backward pass untuk training


