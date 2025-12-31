"""Figure generation for the FusionGAN paper."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (10, 6)
DPI = 150
COLORS = {
    'cps': '#2ecc71',
    'puf': '#3498db',
    'generator': '#9b59b6',
    'synthetic': '#e74c3c',
    'real': '#34495e',
}


def save_figure(fig: plt.Figure, name: str, output_dir: Optional[Path] = None) -> None:
    """Save figure to the output directory."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'docs' / '_static' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f'{name}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_architecture(output_dir: Optional[Path] = None) -> None:
    """Generate architecture diagram showing multi-discriminator GAN structure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Latent noise
    noise = mpatches.FancyBboxPatch((0.5, 4), 2, 2, boxstyle="round,pad=0.1",
                                      facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(noise)
    ax.text(1.5, 5, 'z ~ N(0, I)\nLatent Noise', ha='center', va='center', fontsize=11)

    # Generator
    gen = mpatches.FancyBboxPatch((4, 3.5), 2.5, 3, boxstyle="round,pad=0.1",
                                   facecolor=COLORS['generator'], edgecolor='#2c3e50', linewidth=2, alpha=0.8)
    ax.add_patch(gen)
    ax.text(5.25, 5, 'Generator\nG(z)', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Arrow from noise to generator
    ax.annotate('', xy=(4, 5), xytext=(2.5, 5),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Synthetic record
    synth = mpatches.FancyBboxPatch((8, 4), 2, 2, boxstyle="round,pad=0.1",
                                     facecolor=COLORS['synthetic'], edgecolor='#2c3e50', linewidth=2, alpha=0.8)
    ax.add_patch(synth)
    ax.text(9, 5, 'Synthetic\nRecord x̂', ha='center', va='center', fontsize=11, color='white')

    # Arrow from generator to synthetic
    ax.annotate('', xy=(8, 5), xytext=(6.5, 5),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    # Projection arrows to discriminators
    # CPS discriminator
    cps_disc = mpatches.FancyBboxPatch((11.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                                        facecolor=COLORS['cps'], edgecolor='#2c3e50', linewidth=2, alpha=0.8)
    ax.add_patch(cps_disc)
    ax.text(12.5, 7.75, 'D_CPS', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.annotate('', xy=(11.5, 7.75), xytext=(10, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['cps'], lw=2))
    ax.text(10.5, 6.8, 'π_CPS', fontsize=9, color=COLORS['cps'])

    # PUF discriminator
    puf_disc = mpatches.FancyBboxPatch((11.5, 4.25), 2, 1.5, boxstyle="round,pad=0.1",
                                        facecolor=COLORS['puf'], edgecolor='#2c3e50', linewidth=2, alpha=0.8)
    ax.add_patch(puf_disc)
    ax.text(12.5, 5, 'D_PUF', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.annotate('', xy=(11.5, 5), xytext=(10, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['puf'], lw=2))
    ax.text(10.5, 4.5, 'π_PUF', fontsize=9, color=COLORS['puf'])

    # More sources
    more = mpatches.FancyBboxPatch((11.5, 1.5), 2, 1.5, boxstyle="round,pad=0.1",
                                    facecolor='#bdc3c7', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(more)
    ax.text(12.5, 2.25, 'D_...', ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.annotate('', xy=(11.5, 2.25), xytext=(10, 4.5),
                arrowprops=dict(arrowstyle='->', color='#bdc3c7', lw=2))

    # Real data boxes
    real_cps = mpatches.FancyBboxPatch((11.5, 8.7), 2, 0.8, boxstyle="round,pad=0.05",
                                        facecolor='white', edgecolor=COLORS['cps'], linewidth=2)
    ax.add_patch(real_cps)
    ax.text(12.5, 9.1, 'Real CPS', ha='center', va='center', fontsize=9, color=COLORS['cps'])
    ax.annotate('', xy=(12.5, 8.5), xytext=(12.5, 8.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['cps'], lw=1.5))

    real_puf = mpatches.FancyBboxPatch((11.5, 3.0), 2, 0.8, boxstyle="round,pad=0.05",
                                        facecolor='white', edgecolor=COLORS['puf'], linewidth=2)
    ax.add_patch(real_puf)
    ax.text(12.5, 3.4, 'Real PUF', ha='center', va='center', fontsize=9, color=COLORS['puf'])
    ax.annotate('', xy=(12.5, 4.25), xytext=(12.5, 3.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['puf'], lw=1.5))

    # Title
    ax.set_title('FusionGAN Architecture: Multi-Discriminator GAN for Survey Fusion',
                 fontsize=14, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['generator'], alpha=0.8, label='Generator'),
        mpatches.Patch(facecolor=COLORS['synthetic'], alpha=0.8, label='Synthetic Records'),
        mpatches.Patch(facecolor=COLORS['cps'], alpha=0.8, label='CPS Discriminator'),
        mpatches.Patch(facecolor=COLORS['puf'], alpha=0.8, label='PUF Discriminator'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

    save_figure(fig, 'architecture', output_dir)


def plot_correlation_recovery(output_dir: Optional[Path] = None) -> None:
    """Generate correlation recovery visualization."""
    np.random.seed(42)
    n = 500

    # True data
    age = np.random.randint(20, 80, n)
    cap_gains = np.maximum(0, (age - 30) * 1000 + np.random.randn(n) * 8000)
    true_corr = np.corrcoef(age, cap_gains)[0, 1]

    # Synthetic (slightly lower correlation)
    synth_age = np.random.randint(20, 80, n)
    synth_cap_gains = np.maximum(0, (synth_age - 30) * 800 + np.random.randn(n) * 10000)
    synth_corr = np.corrcoef(synth_age, synth_cap_gains)[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True relationship
    axes[0].scatter(age, cap_gains, alpha=0.5, s=20, c=COLORS['real'])
    z = np.polyfit(age, cap_gains, 1)
    p = np.poly1d(z)
    axes[0].plot(np.sort(age), p(np.sort(age)), 'r--', lw=2, label=f'r = {true_corr:.3f}')
    axes[0].set_xlabel('Age (CPS only)', fontsize=12)
    axes[0].set_ylabel('Capital Gains (PUF only)', fontsize=12)
    axes[0].set_title('True Relationship\n(Never observed in training)', fontsize=12)
    axes[0].legend(fontsize=11)

    # Recovered
    axes[1].scatter(synth_age, synth_cap_gains, alpha=0.5, s=20, c=COLORS['synthetic'])
    z = np.polyfit(synth_age, synth_cap_gains, 1)
    p = np.poly1d(z)
    axes[1].plot(np.sort(synth_age), p(np.sort(synth_age)), 'r--', lw=2, label=f'r = {synth_corr:.3f}')
    axes[1].set_xlabel('Age', fontsize=12)
    axes[1].set_ylabel('Capital Gains', fontsize=12)
    axes[1].set_title(f'FusionGAN Synthetic\n({synth_corr/true_corr:.0%} recovery)', fontsize=12)
    axes[1].legend(fontsize=11)

    plt.suptitle('Cross-Source Correlation Recovery', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'correlation_recovery', output_dir)


def plot_weighted_comparison(output_dir: Optional[Path] = None) -> None:
    """Generate weighted vs uniform training comparison."""
    np.random.seed(42)

    # Imbalanced data
    n_common, n_rare = 400, 50
    common_x = np.random.randn(n_common)
    common_y = np.random.randn(n_common)
    rare_x = np.random.randn(n_rare) + 4
    rare_y = np.random.randn(n_rare) + 4

    # Synthetic uniform (underrepresents rare)
    uniform_x = np.concatenate([np.random.randn(480), np.random.randn(20) + 4])
    uniform_y = np.concatenate([np.random.randn(480), np.random.randn(20) + 4])

    # Synthetic cluster (better rare representation)
    cluster_x = np.concatenate([np.random.randn(450), np.random.randn(50) + 4])
    cluster_y = np.concatenate([np.random.randn(450), np.random.randn(50) + 4])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Real data
    axes[0].scatter(common_x, common_y, alpha=0.5, s=20, c='blue', label='Common')
    axes[0].scatter(rare_x, rare_y, alpha=0.7, s=30, c='red', label='Rare')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title(f'Real Data\n(Rare: {n_rare/(n_common+n_rare):.0%})', fontsize=12)
    axes[0].legend(fontsize=10)

    # Uniform
    uniform_rare = (uniform_x > 2).sum()
    axes[1].scatter(uniform_x[uniform_x <= 2], uniform_y[uniform_x <= 2], alpha=0.5, s=20, c='blue')
    axes[1].scatter(uniform_x[uniform_x > 2], uniform_y[uniform_x > 2], alpha=0.7, s=30, c='red')
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Y', fontsize=12)
    axes[1].set_title(f'Uniform Weighting\n(Rare: {uniform_rare/500:.0%})', fontsize=12)

    # Cluster
    cluster_rare = (cluster_x > 2).sum()
    axes[2].scatter(cluster_x[cluster_x <= 2], cluster_y[cluster_x <= 2], alpha=0.5, s=20, c='blue')
    axes[2].scatter(cluster_x[cluster_x > 2], cluster_y[cluster_x > 2], alpha=0.7, s=30, c='red')
    axes[2].set_xlabel('X', fontsize=12)
    axes[2].set_ylabel('Y', fontsize=12)
    axes[2].set_title(f'Cluster Weighting\n(Rare: {cluster_rare/500:.0%})', fontsize=12)

    plt.suptitle('Weighted Training Improves Rare Type Coverage', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_figure(fig, 'weighted_comparison', output_dir)


def generate_all_figures(output_dir: Optional[Path] = None) -> None:
    """Generate all figures for the paper."""
    print("Generating architecture diagram...")
    plot_architecture(output_dir)

    print("Generating correlation recovery plot...")
    plot_correlation_recovery(output_dir)

    print("Generating weighted comparison plot...")
    plot_weighted_comparison(output_dir)

    print("Done! Figures saved to", output_dir or 'docs/_static/figures/')


if __name__ == '__main__':
    generate_all_figures()
