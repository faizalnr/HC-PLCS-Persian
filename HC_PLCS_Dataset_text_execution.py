#!/usr/bin/env python3
"""Reproducible 1,000-sample HC-PLCS evaluation on the AICT Sharif Persian typo dataset.

This harness calls the supplied HC-PLCS encryption/decryption functions without changing
its transformation logic. It records each nonce and ciphertext so runs can be audited.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, platform, sys, time, tracemalloc, urllib.request
from pathlib import Path
from statistics import mean, median, stdev
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hc_plcs_full_experiment as hc

DATA_URL = "https://huggingface.co/datasets/aictsharif/perisan_texts_with_typos/resolve/main/train.csv"
DEFAULT_SEED = 20261027


def get_training_csv(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset: {DATA_URL}")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "HC-PLCS-research-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        out.write(response.read())
    return path


def prepare_text(value: str) -> str:
    # Matches the manuscript's stated preprocessing: canonicalize then remove marks
    # that the HC-PLCS diacritic layer later removes during decryption.
    return hc.remove_diacritics(hc.normalize(str(value)))


def get_peak_measurement(fn):
    tracemalloc.start()
    start = time.perf_counter_ns()
    result = fn()
    elapsed_ns = time.perf_counter_ns() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed_ns / 1e6, peak / 1024


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = get_training_csv(Path(args.train_csv))
    all_rows = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "with_typos" not in all_rows.columns:
        raise ValueError(f"Expected a 'with_typos' column, found: {list(all_rows.columns)}")
    if "fixed" not in all_rows.columns:
        all_rows["fixed"] = ""
    all_rows = all_rows.dropna(subset=["with_typos"]).copy()
    all_rows["with_typos"] = all_rows["with_typos"].astype(str)
    all_rows = all_rows[all_rows["with_typos"].str.strip().ne("")].copy()
    all_rows["source_row_index"] = all_rows.index + 1
    if len(all_rows) < args.samples:
        raise ValueError(f"Requested {args.samples} samples but only {len(all_rows)} eligible training rows are available.")

    selected = all_rows.sample(n=args.samples, replace=False, random_state=args.sampling_seed).copy()
    selected["Plaintext (P)"] = selected["with_typos"].map(prepare_text)
    selected = selected[selected["Plaintext (P)"].str.strip().ne("")].copy()
    selected.insert(0, "No", range(1, len(selected) + 1))
    if len(selected) != args.samples:
        raise ValueError(f"After preprocessing only {len(selected)} rows remain; requested {args.samples}. Choose a larger eligible subset.")
    selected.to_csv(out / "selected_1000_samples.csv", index=False, encoding="utf-8-sig")
    # Key handling follows the supplied implementation. A textual research key is mapped
    # deterministically to 32 bytes by text_to_key; this is NOT a recommendation for production use.
    key = hc.text_to_key(args.key)
    (out / "key_info.txt").write_text(
        "HC-PLCS research evaluation key (test-only; do not use for real security)\n"
        f"Key mode: textual input passed through text_to_key()\nKey length after conversion: {len(key)*8} bits\n"
        f"Key label: {args.key}\nSampling seed: {args.sampling_seed}\nSamples requested: {args.samples}\n",
        encoding="utf-8")

    results = []
    for counter, (_, row) in enumerate(selected.iterrows(), start=1):
        p = row["Plaintext (P)"]
        nonce = hc.generate_nonce()
        (enc_return, enc_ms, enc_peak) = get_peak_measurement(lambda: hc.encrypt(p, key, nonce=nonce))
        core, packet, meta = enc_return
        (recovered, dec_ms, dec_peak) = get_peak_measurement(lambda: hc.decrypt(packet, key))
        p_norm = hc.normalize(p)
        core_bytes = len(core.encode("utf-8"))
        p_bytes = len(p_norm.encode("utf-8"))
        packet_bytes = len(packet)
        inter = meta.get("intermediate", "")
        results.append({
            "No": int(row["No"]), "source_row_index": int(row["source_row_index"]),
            "plaintext": p_norm, "ciphertext": core, "recovered": recovered,
            "correct": int(recovered == p_norm), "nonce_hex": nonce.hex(),
            "packet_base64": base64.b64encode(packet).decode("ascii"),
            "plaintext_chars": len(p_norm), "ciphertext_chars": len(core),
            "plaintext_bytes": p_bytes, "ciphertext_bytes": core_bytes, "packet_bytes": packet_bytes,
            "EER_core": core_bytes / max(1, p_bytes),
            "SO_core_percent": (core_bytes-p_bytes) / max(1, p_bytes) * 100,
            "EER_packet": packet_bytes / max(1, p_bytes),
            "SO_packet_percent": (packet_bytes-p_bytes) / max(1, p_bytes) * 100,
            "entropy_plain": hc.entropy(p_norm), "entropy_cipher": hc.entropy(core),
            "entropy_gain": hc.entropy(core)-hc.entropy(p_norm),
            "chi_square_plain_sample": hc.chi_square_uniform(p_norm),
            "chi_square_cipher_sample": hc.chi_square_uniform(core),
            "latency_encrypt_ms": enc_ms, "latency_decrypt_ms": dec_ms,
            "peak_memory_encrypt_KB": enc_peak, "peak_memory_decrypt_KB": dec_peak,
            "throughput_encrypt_chars_s": len(p_norm) / max(enc_ms/1000, 1e-12),
            "throughput_decrypt_chars_s": len(core) / max(dec_ms/1000, 1e-12),
            "CWDR": hc.wtr_linguistic(p_norm, core),
            "LTR": hc.ltr_from_intermediate(p_norm, inter.split()),
        })
        if counter % 100 == 0:
            print(f"Processed {counter}/{args.samples}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(out / "sample_results.csv", index=False, encoding="utf-8-sig")
    # Primary aggregate statistics are calculated over the complete sampled corpus.
    pcat = "".join(result_df["plaintext"].astype(str))
    ccat = "".join(result_df["ciphertext"].astype(str))
    correct = int(result_df["correct"].sum())
    summary = {
        "Dataset": "AICT Sharif Persian Typo Correction Dataset / train.csv",
        "Train_rows_after_empty_filter": int(len(all_rows)),
        "Samples": int(len(result_df)), "Sampling_seed": int(args.sampling_seed),
        "Key_bits_after_text_to_key": int(len(key)*8),
        "Unique_nonces": int(result_df["nonce_hex"].nunique()),
        "EER_core_mean": float(result_df["EER_core"].mean()),
        "EER_core_sd": float(result_df["EER_core"].std(ddof=1)),
        "Storage_overhead_core_percent_mean": float(result_df["SO_core_percent"].mean()),
        "EER_packet_mean": float(result_df["EER_packet"].mean()),
        "EER_packet_sd": float(result_df["EER_packet"].std(ddof=1)),
        "Storage_overhead_packet_percent_mean": float(result_df["SO_packet_percent"].mean()),
        "CWDR_mean": float(result_df["CWDR"].mean()), "LTR_mean": float(result_df["LTR"].mean()),
        "RA_percent": correct/max(1,len(result_df))*100, "Correctly_recovered": correct,
        "Recovery_failures": int(len(result_df)-correct),
        "Entropy_plain_pooled_bits_per_char": hc.entropy(pcat),
        "Entropy_cipher_pooled_bits_per_char": hc.entropy(ccat),
        "Entropy_gain_pooled_bits_per_char": hc.entropy(ccat)-hc.entropy(pcat),
        "Chi_square_plain_pooled": hc.chi_square_uniform(pcat),
        "Chi_square_cipher_pooled": hc.chi_square_uniform(ccat),
        "PMU_encrypt_KB_mean": float(result_df["peak_memory_encrypt_KB"].mean()),
        "PMU_decrypt_KB_mean": float(result_df["peak_memory_decrypt_KB"].mean()),
        "Latency_encrypt_ms_mean": float(result_df["latency_encrypt_ms"].mean()),
        "Latency_encrypt_ms_median": float(result_df["latency_encrypt_ms"].median()),
        "Latency_encrypt_ms_sd": float(result_df["latency_encrypt_ms"].std(ddof=1)),
        "Latency_decrypt_ms_mean": float(result_df["latency_decrypt_ms"].mean()),
        "Latency_decrypt_ms_median": float(result_df["latency_decrypt_ms"].median()),
        "Latency_decrypt_ms_sd": float(result_df["latency_decrypt_ms"].std(ddof=1)),
        "Throughput_encrypt_chars_s_mean": float(result_df["throughput_encrypt_chars_s"].mean()),
        "Throughput_decrypt_chars_s_mean": float(result_df["throughput_decrypt_chars_s"].mean()),
        "EER_core_median": float(result_df["EER_core"].median()),
        "EER_core_p95": float(result_df["EER_core"].quantile(.95)),
    }
    Path(out / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"Metric": k, "Value": v} for k,v in summary.items()]).to_csv(out/"summary_metrics.csv", index=False, encoding="utf-8-sig")

    # Paired sensitivity experiments use test-only nonces; within each pair the nonce is fixed
    # so that the measured edit distance is not confounded by independent nonce changes.
    # A separate sensitivity key is used to avoid using production-evaluation key/nonces.
    sens_key = hashlib.sha256(b"HC-PLCS-sensitivity-test-only|" + key).digest()
    sens_key2 = bytearray(sens_key); sens_key2[0] ^= 0x01; sens_key2 = bytes(sens_key2)
    ps_vals, ks_vals = [], []
    for i, row in result_df.iterrows():
        p = row["plaintext"]
        if not p:
            continue
        pos = max(0, len(p)//2)
        replacement = "ا" if p[pos] != "ا" else "ب"
        p2 = p[:pos] + replacement + p[pos+1:]
        n_plain = hc.generate_nonce()
        c1 = hc.encrypt(p, sens_key, nonce=n_plain)[0]
        c2 = hc.encrypt(p2, sens_key, nonce=n_plain)[0]
        ps_vals.append(hc.normalized_edit_distance(c1,c2)*100)
        n_key = hc.generate_nonce()
        ck1 = hc.encrypt(p, sens_key, nonce=n_key)[0]
        ck2 = hc.encrypt(p, sens_key2, nonce=n_key)[0]
        ks_vals.append(hc.normalized_edit_distance(ck1,ck2)*100)
    sensitivity = pd.DataFrame([
        {"Experiment":"Plaintext sensitivity", "N":len(ps_vals), "Mean divergence (%)":mean(ps_vals) if ps_vals else 0,
         "SD (percentage points)":stdev(ps_vals) if len(ps_vals)>1 else 0,
         "Protocol":"One central code point changed; same test-only key and nonce within each pair"},
        {"Experiment":"Key sensitivity", "N":len(ks_vals), "Mean divergence (%)":mean(ks_vals) if ks_vals else 0,
         "SD (percentage points)":stdev(ks_vals) if len(ks_vals)>1 else 0,
         "Protocol":"One key bit flipped; same test-only nonce within each pair"},
    ])
    sensitivity.to_csv(out/"sensitivity_results.csv", index=False, encoding="utf-8-sig")

    # Figures for manuscript analysis.
    plt.figure(figsize=(8.6,5.2))
    plt.boxplot([result_df["EER_core"], result_df["EER_packet"]], labels=["Core EER", "Packet EER"], showmeans=True)
    plt.ylabel("Expansion ratio (×)"); plt.title("HC-PLCS Ciphertext Expansion — 1,000 Persian Samples")
    plt.grid(axis="y", linestyle=":", alpha=.4); plt.tight_layout(); plt.savefig(out/"Figure1_EER_distribution.png", dpi=300); plt.close()

    plt.figure(figsize=(8.6,5.2))
    plt.boxplot([result_df["entropy_plain"], result_df["entropy_cipher"]], labels=["Plaintext", "Ciphertext"], showmeans=True)
    plt.ylabel("Character entropy (bits/character)"); plt.title("Per-Sample Character Entropy Distribution")
    plt.grid(axis="y", linestyle=":", alpha=.4); plt.tight_layout(); plt.savefig(out/"Figure2_Entropy_distribution.png", dpi=300); plt.close()

    plt.figure(figsize=(8.6,5.2))
    plt.boxplot([result_df["latency_encrypt_ms"], result_df["latency_decrypt_ms"]], labels=["Encryption", "Decryption"], showmeans=True)
    plt.ylabel("Latency (ms/sample)"); plt.title("HC-PLCS Runtime Distribution — 1,000 Samples")
    plt.grid(axis="y", linestyle=":", alpha=.4); plt.tight_layout(); plt.savefig(out/"Figure3_Latency_distribution.png", dpi=300); plt.close()

    p_counts = pd.Series(list(pcat)).value_counts(); c_counts = pd.Series(list(ccat)).value_counts()
    chars = list((p_counts.add(c_counts, fill_value=0)).sort_values(ascending=False).head(20).index)
    freq = pd.DataFrame({"Character":chars, "Plaintext frequency":[int(p_counts.get(c,0)) for c in chars],
                         "Ciphertext frequency":[int(c_counts.get(c,0)) for c in chars]})
    freq.to_csv(out/"character_frequency_top20.csv", index=False, encoding="utf-8-sig")
    x = np.arange(len(chars)); width = .38
    plt.figure(figsize=(12,5.8))
    plt.bar(x-width/2, freq["Plaintext frequency"], width, label="Plaintext")
    plt.bar(x+width/2, freq["Ciphertext frequency"], width, label="Ciphertext")
    plt.xticks(x, chars); plt.xlabel("Character"); plt.ylabel("Frequency")
    plt.title("Top-20 Character Frequencies: Plaintext vs. HC-PLCS Ciphertext"); plt.legend(frameon=False)
    plt.tight_layout(); plt.savefig(out/"Figure4_Character_frequency.png", dpi=300); plt.close()

    system_info = {
        "Python":sys.version, "Platform":platform.platform(), "Processor":platform.processor(),
        "pandas":pd.__version__, "numpy":np.__version__, "matplotlib":plt.matplotlib.__version__,
        "Implementation file":"hc_plcs_full_experiment.py",
    }
    Path(out/"environment.json").write_text(json.dumps(system_info, indent=2), encoding="utf-8")
    # One Excel workbook for manuscript-ready audit; all raw sample outcomes remain available in CSV too.
    with pd.ExcelWriter(out/"HC_PLCS_AICT_1000_Results.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([{"Metric":k,"Value":v} for k,v in summary.items()]).to_excel(writer, sheet_name="Summary", index=False)
        sensitivity.to_excel(writer, sheet_name="Sensitivity", index=False)
        result_df.drop(columns=["packet_base64"]).to_excel(writer, sheet_name="SampleResults", index=False)
        selected[["No","source_row_index","with_typos","fixed","Plaintext (P)"]].to_excel(writer, sheet_name="SelectedSamples", index=False)
        freq.to_excel(writer, sheet_name="CharacterFrequency", index=False)

    # Compact report in text form.
    report = [
        "HC-PLCS — AICT Sharif Persian Typo Correction Dataset: 1,000-sample evaluation",
        f"Training rows after empty filtering: {len(all_rows)}", f"Selected samples: {len(result_df)}",
        f"Sampling seed: {args.sampling_seed}", f"Unique nonces: {summary['Unique_nonces']}/{len(result_df)}",
        f"Exact recovery: {correct}/{len(result_df)} ({summary['RA_percent']:.2f}%)",
        f"Core EER: {summary['EER_core_mean']:.6f} ± {summary['EER_core_sd']:.6f}",
        f"Core storage overhead (mean per sample): {summary['Storage_overhead_core_percent_mean']:.4f}%",
        f"Packet EER: {summary['EER_packet_mean']:.6f} ± {summary['EER_packet_sd']:.6f}",
        f"Packet storage overhead (mean per sample): {summary['Storage_overhead_packet_percent_mean']:.4f}%",
        f"Pooled plaintext entropy: {summary['Entropy_plain_pooled_bits_per_char']:.6f} bits/char",
        f"Pooled ciphertext entropy: {summary['Entropy_cipher_pooled_bits_per_char']:.6f} bits/char",
        f"Pooled entropy gain: {summary['Entropy_gain_pooled_bits_per_char']:+.6f} bits/char",
        f"Pooled chi-square (plaintext): {summary['Chi_square_plain_pooled']:.6f}",
        f"Pooled chi-square (ciphertext): {summary['Chi_square_cipher_pooled']:.6f}",
        f"Mean encryption latency: {summary['Latency_encrypt_ms_mean']:.6f} ms/sample",
        f"Mean decryption latency: {summary['Latency_decrypt_ms_mean']:.6f} ms/sample",
        f"Mean encryption peak memory: {summary['PMU_encrypt_KB_mean']:.3f} KB",
        f"Mean decryption peak memory: {summary['PMU_decrypt_KB_mean']:.3f} KB",
        f"Plaintext sensitivity: {sensitivity.iloc[0]['Mean divergence (%)']:.4f} ± {sensitivity.iloc[0]['SD (percentage points)']:.4f}% (N={int(sensitivity.iloc[0]['N'])})",
        f"Key sensitivity: {sensitivity.iloc[1]['Mean divergence (%)']:.4f} ± {sensitivity.iloc[1]['SD (percentage points)']:.4f}% (N={int(sensitivity.iloc[1]['N'])})",
        "",
        "Interpretation caution: entropy and chi-square are descriptive statistics, not security proofs.",
        "The current implementation does not implement an authentication tag despite TAG_BYTES being defined.",
        "Check recovery failures in sample_results.csv before making a reversibility claim.",
        "Timing values are environment-specific and include Python/tracemalloc overhead.",
    ]
    (out/"HC_PLCS_AICT_1000_Report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nResults saved under: {out.resolve()}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="train.csv", help="Local HF train.csv; download is attempted if absent")
    ap.add_argument("--output-dir", default="HC_PLCS_AICT_1000_Results")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--sampling-seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--key", default="advanced_key", help="Research key text, converted by supplied implementation; test only")
    args=ap.parse_args(); run(args)

if __name__ == "__main__":
    main()
