"""
Testing dan Validasi untuk Implementasi Transformer
Menguji semua komponen dan memverifikasi dimensi, masking, dan output
"""

import numpy as np
from transformer import (
    TokenEmbedding,
    PositionalEncoding,
    ScaledDotProductAttention,
    MultiHeadAttention,
    FeedForwardNetwork,
    LayerNormalization,
    TransformerBlock,
    DecoderOnlyTransformer,
    create_causal_mask,
    softmax
)


def test_token_embedding():
    """Test Token Embedding"""
    print("=" * 60)
    print("Testing Token Embedding")
    print("=" * 60)

    vocab_size = 100
    d_model = 64
    batch_size = 2
    seq_len = 10

    token_emb = TokenEmbedding(vocab_size, d_model)
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    embeddings = token_emb.forward(token_ids)

    print(f"Input shape: {token_ids.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    assert embeddings.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    print("[PASS] Token Embedding test passed!\n")


def test_positional_encoding():
    """Test Positional Encoding"""
    print("=" * 60)
    print("Testing Positional Encoding")
    print("=" * 60)

    max_seq_len = 100
    d_model = 64
    batch_size = 2
    seq_len = 10

    pos_enc = PositionalEncoding(max_seq_len, d_model)
    embeddings = np.random.randn(batch_size, seq_len, d_model)

    output = pos_enc.forward(embeddings)

    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    print("[PASS] Positional Encoding test passed!\n")


def test_causal_mask():
    """Test Causal Masking"""
    print("=" * 60)
    print("Testing Causal Mask")
    print("=" * 60)

    seq_len = 5
    mask = create_causal_mask(seq_len)

    print(f"Causal mask shape: {mask.shape}")
    print(f"Expected shape: ({seq_len}, {seq_len})")
    print("\nCausal mask (0 = allowed, -inf = masked):")
    print(np.where(mask == 0, 0, 1))  # Print 0/1 untuk readability

    # Verify diagonal dan bagian bawah adalah 0 (allowed)
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i, j] == 0, f"Position ({i},{j}) should be allowed!"
            else:
                assert mask[i, j] < -1e8, f"Position ({i},{j}) should be masked!"

    print("[PASS] Causal Mask test passed!\n")


def test_scaled_dot_product_attention():
    """Test Scaled Dot-Product Attention"""
    print("=" * 60)
    print("Testing Scaled Dot-Product Attention")
    print("=" * 60)

    batch_size = 2
    seq_len = 5
    d_k = 64

    attention = ScaledDotProductAttention(d_k)

    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)
    mask = create_causal_mask(seq_len)

    output, attn_weights = attention.forward(Q, K, V, mask)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_k})")
    print(f"Expected attention shape: ({batch_size}, {seq_len}, {seq_len})")

    assert output.shape == (batch_size, seq_len, d_k), "Output shape mismatch!"
    assert attn_weights.shape == (batch_size, seq_len, seq_len), "Attention shape mismatch!"

    # Test softmax sum = 1
    print(f"\nAttention weights sum per row: {np.sum(attn_weights[0], axis=-1)}")
    assert np.allclose(np.sum(attn_weights, axis=-1), 1.0), "Softmax sum should be 1!"

    print("[PASS] Scaled Dot-Product Attention test passed!\n")


def test_multi_head_attention():
    """Test Multi-Head Attention"""
    print("=" * 60)
    print("Testing Multi-Head Attention")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    batch_size = 2
    seq_len = 10

    mha = MultiHeadAttention(d_model, num_heads)
    x = np.random.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)

    output, attn_weights = mha.forward(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Expected attention shape: ({batch_size}, {num_heads}, {seq_len}, {seq_len})")

    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch!"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention shape mismatch!"

    print("[PASS] Multi-Head Attention test passed!\n")


def test_feed_forward_network():
    """Test Feed-Forward Network"""
    print("=" * 60)
    print("Testing Feed-Forward Network")
    print("=" * 60)

    d_model = 64
    d_ff = 256
    batch_size = 2
    seq_len = 10

    ffn = FeedForwardNetwork(d_model, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)

    output = ffn.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")

    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    print("[PASS] Feed-Forward Network test passed!\n")


def test_layer_normalization():
    """Test Layer Normalization"""
    print("=" * 60)
    print("Testing Layer Normalization")
    print("=" * 60)

    d_model = 64
    batch_size = 2
    seq_len = 10

    ln = LayerNormalization(d_model)
    x = np.random.randn(batch_size, seq_len, d_model)

    output = ln.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")

    # Check normalization: mean ~ 0, variance ~ 1
    mean = np.mean(output, axis=-1)
    var = np.var(output, axis=-1)

    print(f"\nOutput mean (should be ~ 0): {mean[0, 0]:.6f}")
    print(f"Output variance (should be ~ 1): {var[0, 0]:.6f}")

    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    assert np.allclose(mean, 0, atol=1e-5), "Mean should be close to 0!"
    assert np.allclose(var, 1, atol=1e-5), "Variance should be close to 1!"

    print("[PASS] Layer Normalization test passed!\n")


def test_transformer_block():
    """Test Transformer Block"""
    print("=" * 60)
    print("Testing Transformer Block")
    print("=" * 60)

    d_model = 64
    num_heads = 8
    d_ff = 256
    batch_size = 2
    seq_len = 10

    block = TransformerBlock(d_model, num_heads, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    mask = create_causal_mask(seq_len)

    output, attn_weights = block.forward(x, mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")

    assert output.shape == (batch_size, seq_len, d_model), "Shape mismatch!"
    print("[PASS] Transformer Block test passed!\n")


def test_full_transformer():
    """Test Full Decoder-Only Transformer"""
    print("=" * 60)
    print("Testing Full Decoder-Only Transformer")
    print("=" * 60)

    # Model parameters
    vocab_size = 100
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 4
    max_seq_len = 50

    # Input
    batch_size = 2
    seq_len = 10
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Create model
    model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len)

    # Forward pass
    logits, probs, attn_weights_list = model.forward(token_ids, return_attention=True)

    print(f"Input shape: {token_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")
    print(f"Expected logits shape: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"Expected probs shape: ({batch_size}, {vocab_size})")
    print(f"Number of layers: {len(attn_weights_list)}")

    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape mismatch!"
    assert probs.shape == (batch_size, vocab_size), "Probs shape mismatch!"
    assert len(attn_weights_list) == num_layers, "Number of attention weights mismatch!"

    # Test probability sums to 1
    prob_sum = np.sum(probs, axis=-1)
    print(f"\nProbability sum per batch: {prob_sum}")
    assert np.allclose(prob_sum, 1.0), "Probabilities should sum to 1!"

    # Show top-5 predictions
    print("\nTop-5 predicted tokens for batch 0:")
    top5_indices = np.argsort(probs[0])[::-1][:5]
    for i, idx in enumerate(top5_indices, 1):
        print(f"  {i}. Token {idx}: probability = {probs[0, idx]:.6f}")

    print("\n[PASS] Full Transformer test passed!\n")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS FOR TRANSFORMER IMPLEMENTATION")
    print("=" * 60 + "\n")

    test_token_embedding()
    test_positional_encoding()
    test_causal_mask()
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_feed_forward_network()
    test_layer_normalization()
    test_transformer_block()
    test_full_transformer()

    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    run_all_tests()
