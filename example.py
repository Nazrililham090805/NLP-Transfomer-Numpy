"""
Demonstrasi cara menggunakan model untuk prediksi token
"""

import numpy as np
from transformer import DecoderOnlyTransformer


def example_basic_usage():
    """Contoh penggunaan dasar model"""
    print("=" * 70)
    print("CONTOH PENGGUNAAN BASIC")
    print("=" * 70)

    # Konfigurasi model kecil untuk demo
    vocab_size = 50        # Ukuran vocabulary kecil
    d_model = 64          # Dimensi embedding
    num_heads = 4         # Jumlah attention heads
    d_ff = 256           # Dimensi FFN
    num_layers = 2       # Jumlah transformer blocks
    max_seq_len = 20     # Panjang maksimum sequence

    print(f"\nKonfigurasi Model:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Attention heads: {num_heads}")
    print(f"  - FFN dimension: {d_ff}")
    print(f"  - Number of layers: {num_layers}")

    # Inisialisasi model
    print("\nInisialisasi model...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    print("Model berhasil dibuat!")

    # Input sequence (contoh: "hello world")
    # Dalam praktik, ini hasil dari tokenizer
    batch_size = 1
    seq_len = 5
    input_tokens = np.array([[2, 15, 8, 23, 42]])  # Token IDs

    print(f"\nInput:")
    print(f"  - Shape: {input_tokens.shape}")
    print(f"  - Token IDs: {input_tokens[0].tolist()}")

    # Forward pass
    print("\nMenjalankan forward pass...")
    logits, probs = model.forward(input_tokens)

    print(f"\nOutput:")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Probabilities shape: {probs.shape}")

    # Prediksi token berikutnya
    next_token_id = np.argmax(probs[0])
    next_token_prob = probs[0, next_token_id]

    print(f"\nPrediksi Token Berikutnya:")
    print(f"  - Token ID: {next_token_id}")
    print(f"  - Probability: {next_token_prob:.6f}")

    # Top-5 predictions
    print(f"\nTop-5 Predictions:")
    top5_indices = np.argsort(probs[0])[::-1][:5]
    for i, token_id in enumerate(top5_indices, 1):
        print(f"  {i}. Token {token_id:2d}: {probs[0, token_id]:.6f}")


def example_with_attention_visualization():
    """Contoh dengan visualisasi attention weights"""
    print("\n" + "=" * 70)
    print("CONTOH DENGAN ATTENTION WEIGHTS")
    print("=" * 70)

    # Konfigurasi
    vocab_size = 30
    d_model = 32
    num_heads = 4
    d_ff = 128
    num_layers = 2
    max_seq_len = 10

    # Inisialisasi model
    print("\nInisialisasi model...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    # Input
    input_tokens = np.array([[5, 12, 3, 8]])

    print(f"\nInput tokens: {input_tokens[0].tolist()}")

    # Forward pass dengan attention weights
    print("\nMenjalankan forward pass dengan attention tracking...")
    logits, probs, attention_weights = model.forward(input_tokens, return_attention=True)

    print(f"\nAttention weights:")
    print(f"  - Number of layers: {len(attention_weights)}")

    # Analisis attention untuk setiap layer
    for layer_idx, attn in enumerate(attention_weights):
        print(f"\n  Layer {layer_idx + 1}:")
        print(f"    - Shape: {attn.shape}")  # [batch, num_heads, seq_len, seq_len]
        print(f"    - Heads: {attn.shape[1]}")

        # Rata-rata attention across heads
        avg_attn = np.mean(attn[0], axis=0)  # Average over heads
        print(f"    - Average attention pattern (first 3 positions):")
        for i in range(min(3, avg_attn.shape[0])):
            attn_str = " ".join([f"{avg_attn[i, j]:.3f}" for j in range(avg_attn.shape[1])])
            print(f"      Position {i}: [{attn_str}]")


def example_batch_processing():
    """Contoh batch processing"""
    print("\n" + "=" * 70)
    print("CONTOH BATCH PROCESSING")
    print("=" * 70)

    # Konfigurasi
    vocab_size = 100
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 3
    max_seq_len = 15

    print("\nInisialisasi model...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    # Batch input dengan sequences berbeda
    batch_size = 3
    seq_len = 6
    batch_input = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    print(f"\nBatch input:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    for i in range(batch_size):
        print(f"  - Sequence {i}: {batch_input[i].tolist()}")

    # Forward pass
    print("\nMenjalankan batch forward pass...")
    logits, probs = model.forward(batch_input)

    print(f"\nOutput:")
    print(f"  - Logits shape: {logits.shape}")
    print(f"  - Probabilities shape: {probs.shape}")

    # Prediksi untuk setiap item dalam batch
    print(f"\nPrediksi token berikutnya untuk setiap sequence:")
    for i in range(batch_size):
        next_token = np.argmax(probs[i])
        next_prob = probs[i, next_token]
        print(f"  Sequence {i}: Token {next_token} (prob: {next_prob:.6f})")


def example_text_generation_simulation():
    """Simulasi text generation (autoregressive)"""
    print("\n" + "=" * 70)
    print("SIMULASI TEXT GENERATION (AUTOREGRESSIVE)")
    print("=" * 70)

    # Konfigurasi
    vocab_size = 50
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2
    max_seq_len = 20

    print("\nInisialisasi model...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )

    # Starting tokens (misalnya: [START] token)
    current_sequence = np.array([[1, 5, 12]])  # Initial prompt

    print(f"\nInitial sequence: {current_sequence[0].tolist()}")
    print("\nGenerating next 7 tokens...")

    # Generate beberapa token (autoregressive)
    num_tokens_to_generate = 7

    for step in range(num_tokens_to_generate):
        # Forward pass
        _, probs = model.forward(current_sequence)

        # Sample dari distribusi (atau ambil argmax untuk greedy)
        # Greedy decoding:
        next_token = np.argmax(probs[0])

        # Tambahkan token ke sequence
        current_sequence = np.concatenate([
            current_sequence,
            np.array([[next_token]])
        ], axis=1)

        print(f"  Step {step + 1}: Generated token {next_token} | "
              f"Sequence: {current_sequence[0].tolist()}")

    print(f"\nFinal generated sequence: {current_sequence[0].tolist()}")
    print(f"Sequence length: {current_sequence.shape[1]}")


def main():
    """Menjalankan semua contoh"""
    print("\n" + "#" * 70)
    print("DEMONSTRASI DECODER-ONLY TRANSFORMER")
    print("#" * 70)

    # Set random seed untuk reproducibility
    np.random.seed(42)

    # Jalankan semua contoh
    example_basic_usage()
    example_with_attention_visualization()
    example_batch_processing()
    example_text_generation_simulation()

    print("\n" + "#" * 70)
    print("SEMUA DEMONSTRASI SELESAI")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
