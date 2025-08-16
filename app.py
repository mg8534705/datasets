import os
import pandas as pd
import torch
import streamlit as st
from transformers import pipeline
import evaluate
from jiwer import wer
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Japanese → English Translation Evaluator", layout="wide")
st.title("🎧 Translation Evaluation App")
st.write("Upload your **validated.tsv** file (from CV corpus) and evaluate translations (Japanese → English).")

# File uploader
uploaded_file = st.file_uploader("Upload validated.tsv", type=["tsv"])

if uploaded_file:
    # Load dataset
    tsv_df = pd.read_csv(uploaded_file, sep="\t")
    st.success(f"Loaded {len(tsv_df)} rows from dataset.")

    # Translator pipeline
    st.info("Loading translation model... (first time may take longer)")
    translator = pipeline("translation", model="staka/fugumt-ja-en", device=0 if torch.cuda.is_available() else -1)

    # Metrics
    bleu_metric = evaluate.load("bleu")

    def compute_latency_metrics(pred_tokens, ref_tokens):
        matches = sum(1 for r, h in zip(ref_tokens, pred_tokens) if r == h)
        la = matches / max(len(ref_tokens), 1)

        delays = []
        for idx, token in enumerate(ref_tokens):
            if token in pred_tokens:
                predicted_idx = pred_tokens.index(token)
                delays.append(abs(predicted_idx - idx))
        atd = sum(delays) / max(len(delays), 1)
        return la, atd

    # Process translations
    metrics_list = []
    progress = st.progress(0)
    for idx, row in tsv_df.iterrows():
        reference_text = row['sentence']
        result = translator(reference_text)
        translated_text = result[0]['translation_text']

        bleu_score = bleu_metric.compute(predictions=[translated_text], references=[[reference_text]])["bleu"]
        wer_score = wer(reference_text, translated_text)
        ref_tokens = reference_text.split()
        pred_tokens = translated_text.split()
        la_score, atd_score = compute_latency_metrics(pred_tokens, ref_tokens)

        metrics_list.append({
            "File": row['path'],
            "Reference": reference_text,
            "Prediction": translated_text,
            "BLEU": bleu_score,
            "WER": wer_score,
            "LA": la_score,
            "ATD": atd_score
        })
        progress.progress((idx + 1) / len(tsv_df))

    metrics_df = pd.DataFrame(metrics_list)

    st.subheader("📊 Translation Metrics")
    st.dataframe(metrics_df)

    # Graphs
    st.subheader("📈 Final Metrics Overview")
    final_metrics = metrics_df[["BLEU", "WER", "LA", "ATD"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars_final = ax.bar(final_metrics.index, final_metrics.values, color=["skyblue","salmon","lightgreen","orange"], edgecolor='black')
    for bar in bars_final:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.05, f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Final Average Metrics")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # PDF Report
    st.subheader("📑 Generate Report")
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Translation Evaluation Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, "This report summarizes BLEU, WER, LA, and ATD metrics "
                              "for the uploaded dataset.\n\n"
                              "Conclusions:\n"
                              "- Higher BLEU = better translations.\n"
                              "- Lower WER = fewer errors.\n"
                              "- LA & ATD provide insights into alignment & latency.\n")

        # Final averages
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Final Average Metrics:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8,
            f"BLEU: {final_metrics['BLEU']:.4f}\n"
            f"WER: {final_metrics['WER']:.4f}\n"
            f"LA: {final_metrics['LA']:.4f}\n"
            f"ATD: {final_metrics['ATD']:.4f}\n"
        )

        report_pdf_path = "translation_report.pdf"
        pdf.output(report_pdf_path)
        with open(report_pdf_path, "rb") as f:
            st.download_button("⬇️ Download Report", f, file_name="translation_report.pdf", mime="application/pdf")
