import time

import torch

from src.selective_ssm.functional import (
    blelloch_parallel_scan,
    naive_sequential_scan,
)


def benchmark_scan_methods(device):
    """Compare different scan implementations across sequence lengths."""

    print(f"Running benchmarks on device: {device}")
    print("=" * 80)

    # Configuration
    B, D, N = 2, 64, 16
    test_configs = [128, 256, 512, 1024, 2048, 4096]

    results = []

    for L in test_configs:
        print(f"\nTesting L={L}...")

        # Generate random data
        A_bar = torch.randn(B, L, D, N, device=device)
        B_bar_x = torch.randn(B, L, D, N, device=device)

        # Warmup
        _ = naive_sequential_scan(A_bar, B_bar_x)
        if device == "cuda":
            torch.cuda.synchronize()

        # Manual recurrence (baseline)
        start = time.time()
        h_manual = torch.zeros(B, D, N, device=device)
        h_manual_all = []
        for t in range(L):
            h_manual = A_bar[:, t] * h_manual + B_bar_x[:, t]
            h_manual_all.append(h_manual.clone())
        h_seq = torch.stack(h_manual_all, dim=1)
        if device == "cuda":
            torch.cuda.synchronize()
        time_sequential = time.time() - start

        # Naive sequential scan
        start = time.time()
        h_naive = naive_sequential_scan(A_bar, B_bar_x)
        if device == "cuda":
            torch.cuda.synchronize()
        time_naive = time.time() - start

        # Blelloch parallel scan
        start = time.time()
        h_blelloch = blelloch_parallel_scan(A_bar, B_bar_x)
        if device == "cuda":
            torch.cuda.synchronize()
        time_blelloch = time.time() - start

        # Verify correctness
        assert torch.allclose(h_seq, h_naive, atol=1e-4, rtol=1e-5), (
            "Naive scan mismatch!"
        )
        assert torch.allclose(h_seq, h_blelloch, atol=1e-4, rtol=1e-5), (
            "Blelloch scan mismatch!"
        )

        # Record results
        results.append(
            {
                "L": L,
                "Sequential (ms)": time_sequential * 1000,
                "Naive (ms)": time_naive * 1000,
                "Blelloch (ms)": time_blelloch * 1000,
                "Speedup (Naive)": time_sequential / time_naive,
                "Speedup (Blelloch)": time_sequential / time_blelloch,
            }
        )

        print(f"  Sequential: {time_sequential * 1000:.2f} ms")
        print(
            f"  Naive scan: {time_naive * 1000:.2f} ms \n"
            f"  (speedup: {time_sequential / time_naive:.2f}x)"
        )
        print(
            f"  Blelloch:   {time_blelloch * 1000:.2f} ms \n"
            f"  (speedup: {time_sequential / time_blelloch:.2f}x)"
        )

    # Print summary table, no pandas
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = (
        f"{'L':>8} | "
        f"{'Sequential (ms)':>18} | "
        f"{'Naive (ms)':>12} | "
        f"{'Blelloch (ms)':>14} | "
        f"{'Speedup (Naive)':>16} | "
        f"{'Speedup (Blelloch)':>18}"
    )
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res['L']:>8} | "
            f"{res['Sequential (ms)']:>18.2f} | "
            f"{res['Naive (ms)']:>12.2f} | "
            f"{res['Blelloch (ms)']:>14.2f} | "
            f"{res['Speedup (Naive)']:>16.2f} | "
            f"{res['Speedup (Blelloch)']:>18.2f}"
        )


if __name__ == "__main__":
    # Run benchmarks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark_scan_methods(device=device)
