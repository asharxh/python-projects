"""
Text Summarization Tool using Hugging Face Transformers (Local, Offline)

Features:
- Summarizes large text files by chunking intelligently with overlap.
- Supports multiple models (default: facebook/bart-large-cnn).
- Uses GPU automatically if available.
- Shows progress with tqdm.
- Friendly CLI with argparse.
- Handles errors gracefully.
- Optionally includes a simple tkinter GUI.

Installation:
- Python 3.7+
- pip install torch transformers tqdm nltk

Usage:
- CLI:
    python summarizer.py --input myfile.txt
    python summarizer.py --input book.txt --output book_summary.txt --model google/pegasus-xsum --max_length 250

- GUI:
    python summarizer.py --gui

"""

import os
import sys
import argparse
import time
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import textwrap

# Optional GUI imports
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# For tokenization and chunking
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

def read_file(file_path):
    """Read text from a file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            raise ValueError("Input file is empty.")
        return text
    except Exception as e:
        raise IOError(f"Error reading file: {e}")

def split_text(text, tokenizer, max_chunk_tokens=3000, overlap_tokens=200):
    """
    Split text into chunks of max_chunk_tokens with overlap.
    Uses sentence tokenization to avoid breaking sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.tokenize(sentence))
        if current_length + sentence_tokens > max_chunk_tokens:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap sentences
            if overlap_tokens > 0:
                # Add overlap sentences from end of current chunk
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(tokenizer.tokenize(sent))
                    if overlap_length + sent_tokens > overlap_tokens:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_length += sent_tokens
                current_chunk = overlap_sentences.copy()
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_text(text, model_name, max_length, device):
    """
    Summarize a single text chunk using the specified model.
    """
    try:
        summarizer = pipeline("summarization", model=model_name, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    # Hugging Face summarization pipeline expects max_length in tokens, but user input is words.
    # We'll approximate tokens ~ words here for max_length.
    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        return summary.strip()
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {e}")

def combine_summaries(summaries):
    """
    Combine multiple summaries into one text.
    """
    return "\n\n".join(summaries)

def save_summary(summary, output_path):
    """
    Save the summary text to a file.
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    except Exception as e:
        raise IOError(f"Failed to save summary: {e}")

def print_friendly_message(message):
    print(message)

def detect_device():
    """
    Detect if GPU is available and return device index for transformers pipeline.
    Returns -1 for CPU, 0 or other int for GPU device.
    """
    if torch.cuda.is_available():
        return 0
    else:
        return -1

def process_summarization(input_path, output_path, model_name, max_length_words):
    start_time = time.time()
    print_friendly_message("📄 Reading file...")
    text = read_file(input_path)

    print_friendly_message("⚙️ Loading tokenizer and model (this may take a moment)...")
    device = detect_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Convert max_length from words to tokens approx (1.3 tokens per word approx)
    max_length_tokens = int(max_length_words * 1.3)
    chunk_size = 800  # max tokens per chunk (BART max is 1024, leave buffer)
    overlap = 100      # tokens overlap

    print_friendly_message("🔍 Splitting text into chunks...")
    chunks = split_text(text, tokenizer, max_chunk_tokens=chunk_size, overlap_tokens=overlap)

    print_friendly_message(f"📝 Summarizing {len(chunks)} chunk(s) (please wait)...")
    summaries = []
    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        summary = summarize_text(chunk, model_name, max_length=max_length_tokens, device=device)
        summaries.append(summary)

    # Combine summaries
    combined_summary = combine_summaries(summaries)

    # Optionally summarize combined summary again if multiple chunks
    if len(summaries) > 1:
        print_friendly_message("🔄 Summarizing combined summary for final concise output...")
        combined_summary = summarize_text(combined_summary, model_name, max_length=max_length_tokens, device=device)

    print_friendly_message("✅ Summary complete! Saving output...")
    save_summary(combined_summary, output_path)

    print_friendly_message("\n--- Summary ---\n")
    print(textwrap.fill(combined_summary, width=80))
    print("\n----------------\n")

    elapsed = time.time() - start_time
    print_friendly_message(f"⏱️ Total processing time: {elapsed:.2f} seconds")

def run_cli():
    parser = argparse.ArgumentParser(description="Local Text Summarization using Hugging Face Transformers")
    parser.add_argument('--input', type=str, required=False, help="Path to input text file")
    parser.add_argument('--output', type=str, default="summary.txt", help="Path to save summary (default: summary.txt)")
    parser.add_argument('--model', type=str, default="facebook/bart-large-cnn", help="Model name (default: facebook/bart-large-cnn)")
    parser.add_argument('--max_length', type=int, default=250, help="Desired summary length in words (default: 250)")
    parser.add_argument('--gui', action='store_true', help="Launch simple GUI (if available)")

    args = parser.parse_args()

    if args.gui:
        if not GUI_AVAILABLE:
            print("Tkinter GUI is not available in your Python environment.")
            sys.exit(1)
        launch_gui()
        return

    if not args.input:
        parser.print_help()
        print("\nError: --input argument is required unless --gui is used.")
        sys.exit(1)

    try:
        process_summarization(args.input, args.output, args.model, args.max_length)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

def launch_gui():
    """
    Simple tkinter GUI for file selection and summarization.
    """
    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            input_entry.delete(0, tk.END)
            input_entry.insert(0, file_path)

    def summarize_action():
        input_path = input_entry.get()
        output_path = output_entry.get() or "summary.txt"
        model_name = model_entry.get() or "facebook/bart-large-cnn"
        try:
            max_length = int(max_length_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Max length must be an integer.")
            return

        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid input file.")
            return

        summarize_button.config(state=tk.DISABLED)
        summary_text.delete(1.0, tk.END)
        summary_text.insert(tk.END, "⚙️ Summarizing (please wait)...\n")
        root.update()

        try:
            start_time = time.time()
            text = read_file(input_path)
            device = detect_device()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            max_length_tokens = int(max_length * 1.3)
            chunk_size = 800
            overlap = 100
            chunks = split_text(text, tokenizer, max_chunk_tokens=chunk_size, overlap_tokens=overlap)
            summaries = []
            for chunk in chunks:
                summary = summarize_text(chunk, model_name, max_length=max_length_tokens, device=device)
                summaries.append(summary)
            combined_summary = combine_summaries(summaries)
            if len(summaries) > 1:
                combined_summary = summarize_text(combined_summary, model_name, max_length=max_length_tokens, device=device)
            save_summary(combined_summary, output_path)
            elapsed = time.time() - start_time

            summary_text.delete(1.0, tk.END)
            summary_text.insert(tk.END, combined_summary + "\n\n")
            summary_text.insert(tk.END, f"✅ Summary complete! Saved as {output_path}\n")
            summary_text.insert(tk.END, f"⏱️ Total processing time: {elapsed:.2f} seconds\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            summarize_button.config(state=tk.NORMAL)

    root = tk.Tk()
    root.title("Text Summarizer")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text="Input File:").grid(row=0, column=0, sticky=tk.W)
    input_entry = tk.Entry(frame, width=50)
    input_entry.grid(row=0, column=1, sticky=tk.W)
    browse_button = tk.Button(frame, text="Browse", command=select_file)
    browse_button.grid(row=0, column=2, padx=5)

    tk.Label(frame, text="Output File:").grid(row=1, column=0, sticky=tk.W)
    output_entry = tk.Entry(frame, width=50)
    output_entry.insert(0, "summary.txt")
    output_entry.grid(row=1, column=1, sticky=tk.W)

    tk.Label(frame, text="Model Name:").grid(row=2, column=0, sticky=tk.W)
    model_entry = tk.Entry(frame, width=50)
    model_entry.insert(0, "facebook/bart-large-cnn")
    model_entry.grid(row=2, column=1, sticky=tk.W)

    tk.Label(frame, text="Max Summary Length (words):").grid(row=3, column=0, sticky=tk.W)
    max_length_entry = tk.Entry(frame, width=10)
    max_length_entry.insert(0, "250")
    max_length_entry.grid(row=3, column=1, sticky=tk.W)

    summarize_button = tk.Button(frame, text="Summarize", command=summarize_action)
    summarize_button.grid(row=4, column=0, columnspan=3, pady=10)

    summary_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
    summary_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    root.mainloop()

if __name__ == "__main__":
    run_cli()
