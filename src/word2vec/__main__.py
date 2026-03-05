"""Run Word2Vec training on Text8 dataset: python -m word2vec"""

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from word2vec.model import SkipGramNS
from word2vec.preprocessing import (
    apply_subsampling,
    build_negative_sampling_table,
    build_vocab,
    generate_training_pairs,
    subsample_probs,
    tokenize,
)
from word2vec.training import train

app = typer.Typer(rich_markup_mode="rich")
console = Console()


def load_text8(data_dir: str = "data") -> str:
    """Download and load the Text8 dataset."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    text_file = data_path / "text8"

    if not text_file.exists():
        zip_path = data_path / "text8.zip"
        if not zip_path.exists():
            console.print("[yellow]Downloading text8 dataset...[/yellow]")
            urlretrieve("http://mattmahoney.net/dc/text8.zip", str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_path)

    return text_file.read_text()


def show_neighbors(model: SkipGramNS, vocab: object, query_words: list[str]) -> None:
    """Display nearest neighbors as a rich table."""
    table = Table(title="Nearest Neighbors", show_lines=True)
    table.add_column("Word", style="bold cyan")
    table.add_column("Top 5 Neighbors", style="green")

    norms = np.linalg.norm(model.w_in, axis=1, keepdims=True) + 1e-10
    normalized = model.w_in / norms

    for word in query_words:
        if word not in vocab.word2id:
            continue
        wid = vocab.word2id[word]
        v = normalized[wid]
        sims = normalized @ v
        sims[wid] = -1
        top_ids = np.argsort(sims)[::-1][:5]
        neighbors = ", ".join(f"{vocab.id2word[i]} ({sims[i]:.3f})" for i in top_ids)
        table.add_row(word, neighbors)

    console.print(table)


@app.command()
def main(
    max_tokens: int = typer.Option(0, help="Limit tokens (0 = full dataset)"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    dim: int = typer.Option(100, help="Embedding dimension"),
    window: int = typer.Option(5, help="Context window size"),
    neg_samples: int = typer.Option(5, help="Number of negative samples"),
    lr: float = typer.Option(0.025, help="Starting learning rate"),
    min_count: int = typer.Option(5, help="Minimum word frequency"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """[bold green]Word2Vec[/bold green] Skip-Gram with Negative Sampling in pure NumPy."""
    console.print(
        Panel.fit(
            "[bold cyan]Word2Vec[/bold cyan] Skip-Gram with Negative Sampling\n"
            "[dim]Pure NumPy implementation[/dim]",
            border_style="bright_blue",
        )
    )

    # Load and preprocess
    with console.status("[bold yellow]Loading text8...[/bold yellow]"):
        text = load_text8()
        tokens = tokenize(text)
        if max_tokens > 0:
            tokens = tokens[:max_tokens]

    rprint(f"  [bold]Tokens:[/bold] {len(tokens):,}")

    with console.status("[bold yellow]Building vocabulary...[/bold yellow]"):
        vocab = build_vocab(tokens, min_count=min_count)

    rprint(f"  [bold]Vocabulary:[/bold] {len(vocab.words):,} words")

    # Subsampling
    rng = np.random.default_rng(seed)
    token_ids = np.array([vocab.word2id[w] for w in tokens if w in vocab.word2id])
    discard_probs = subsample_probs(vocab.freqs, t=1e-5)
    token_ids = apply_subsampling(token_ids, discard_probs, rng)
    rprint(f"  [bold]After subsampling:[/bold] {len(token_ids):,} tokens")

    # Training pairs
    with console.status("[bold yellow]Generating training pairs...[/bold yellow]"):
        pairs = generate_training_pairs(token_ids, window_size=window)

    rprint(f"  [bold]Training pairs:[/bold] {len(pairs):,}")

    # Build model and train
    neg_table = build_negative_sampling_table(vocab.freqs)
    model = SkipGramNS(vocab_size=len(vocab.words), embedding_dim=dim, seed=seed)

    console.print("\n[bold bright_green]Training...[/bold bright_green]")
    loss_history = train(
        model, pairs, neg_table, epochs=epochs, neg_samples=neg_samples, lr_start=lr
    )

    # Results
    console.print()
    loss_table = Table(title="Training Summary", show_lines=True)
    loss_table.add_column("Epoch", style="bold")
    loss_table.add_column("Loss", style="yellow")
    for i, loss in enumerate(loss_history):
        loss_table.add_row(str(i + 1), f"{loss:.4f}")
    console.print(loss_table)

    console.print()
    query_words = ["king", "queen", "computer", "water", "one", "good", "city", "war"]
    show_neighbors(model, vocab, query_words)


if __name__ == "__main__":
    app()
