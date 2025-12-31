---
kernelspec:
  name: python3
  display_name: Python 3
---

# Technical Appendix

## A. Network Architectures

### Generator

```
Input: z ∈ ℝ³² (latent noise)
│
├── Linear(32 → 128) + ReLU
├── Linear(128 → 128) + ReLU
├── Linear(128 → 128) + ReLU
├── Linear(128 → |V|)
│
Output: x̂ ∈ ℝ^|V| (synthetic record)
```

### Discriminator (per source)

```
Input: x ∈ ℝ^|Vₖ| (projected record)
│
├── Linear(|Vₖ| → 128) + LeakyReLU(0.2)
├── Linear(128 → 128) + LeakyReLU(0.2)
├── Linear(128 → 64) + LeakyReLU(0.2)
├── Linear(64 → 1) + Sigmoid
│
Output: p ∈ [0, 1] (probability of real)
```

## B. Training Algorithm

```{prf:algorithm} FusionGAN Training
:label: alg-fusiongan

**Input:** Sources $\{S_1, \ldots, S_K\}$, epochs $T$, batch size $B$

**Initialize:** Generator $G$, Discriminators $\{D_1, \ldots, D_K\}$

**for** $t = 1$ to $T$ **do**
    **for** each source $S_k$ **do**
        Sample batch $\{x_i\}_{i=1}^B$ from $S_k$
        Sample noise $\{z_i\}_{i=1}^B \sim \mathcal{N}(0, I)$

        // Update discriminator
        $\hat{x} \leftarrow G(z)$
        $\mathcal{L}_{D_k} \leftarrow -\frac{1}{B}\sum_i [w_i \log D_k(x_i) + \log(1 - D_k(\pi_{V_k}(\hat{x}_i)))]$
        $\theta_{D_k} \leftarrow \theta_{D_k} - \eta_D \nabla \mathcal{L}_{D_k}$

        // Update generator
        $\hat{x} \leftarrow G(z)$
        $\mathcal{L}_G \leftarrow -\frac{1}{B}\sum_i \log D_k(\pi_{V_k}(\hat{x}_i))$
        $\theta_G \leftarrow \theta_G - \eta_G \nabla \mathcal{L}_G$
    **end for**
**end for**

**return** $G$
```

## C. Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latent dimension | 32 | Size of noise vector |
| Hidden dimension | 128 | Width of hidden layers |
| Generator learning rate | 1e-4 | Adam optimizer |
| Discriminator learning rate | 1e-4 | Adam optimizer |
| Adam betas | (0.5, 0.999) | Momentum parameters |
| Batch size | 64 | Training batch size |
| Holdout fraction | 0.2 | For evaluation |
| n_critic | 1 | D updates per G update |

## D. Weighting Strategies

### Uniform Weighting

$$w(x) = 1$$

All samples contribute equally to the discriminator loss.

### Cluster-Based Weighting

1. Fit k-means with $k = 20$ clusters
2. Assign each sample to cluster $c(x)$
3. Weight inversely by cluster size:

$$w(x) = \frac{n}{|c(x)| \cdot k}$$

where $n$ is total sample size and $|c(x)|$ is the size of sample $x$'s cluster.

### Density-Based Weighting

1. Standardize features to zero mean, unit variance
2. Fit kernel density estimator with bandwidth 0.5
3. Weight inversely by density:

$$w(x) = \frac{1}{\hat{p}(x) + \epsilon}$$

4. Clip weights at 99th percentile to avoid extreme values
5. Normalize weights to sum to $n$

## E. Coverage Metric Details

The coverage metric measures how well synthetic data covers the real distribution:

$$\text{Coverage}_k(H, S) = \frac{1}{|H|} \sum_{x \in H} \min_{i=1}^k d(x, S_i^{(k)})$$

where $S_i^{(k)}$ is the $i$-th nearest neighbor of $x$ in $S$ and $d$ is Euclidean distance.

For $k = 1$, this reduces to mean nearest-neighbor distance.

**Interpretation:**
- Coverage = 0: Perfect coverage (synthetic exactly matches real)
- Coverage > 0: Average distance from real points to nearest synthetic
- Lower is better

## F. MMD with Median Heuristic

The Maximum Mean Discrepancy with RBF kernel:

$$\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x, x')] + \mathbb{E}_{y,y' \sim Q}[k(y, y')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)]$$

where $k(x, y) = \exp(-\gamma \|x - y\|^2)$.

**Median heuristic for bandwidth selection:**

$$\gamma = \frac{1}{2 \cdot \text{median}(\{d(x_i, x_j) : i \neq j\})^2}$$

This automatically adapts to the scale of the data.

## G. Relationship to Statistical Matching

Traditional statistical matching {cite:p}`rodgers1984` assumes **conditional independence**:

$$p(V_A \setminus V_B, V_B \setminus V_A | V_A \cap V_B) = p(V_A \setminus V_B | V_A \cap V_B) \cdot p(V_B \setminus V_A | V_A \cap V_B)$$

where $V_A$ and $V_B$ are variables in sources A and B.

FusionGAN makes no such assumption. Instead, it learns the joint distribution implicitly through adversarial training. The generator must produce records that fool all discriminators simultaneously, which requires learning correct correlations across all variable pairs.

## H. Computational Complexity

**Per-epoch complexity:**

- Generator forward: $O(B \cdot d \cdot h)$ where $B$ = batch size, $d$ = output dim, $h$ = hidden dim
- Each discriminator: $O(B \cdot |V_k| \cdot h)$
- Total per source: $O(B \cdot (d + |V_k|) \cdot h)$
- Total per epoch: $O(K \cdot n \cdot (d + \bar{|V|}) \cdot h / B)$

where $K$ = number of sources, $n$ = samples per source, $\bar{|V|}$ = average variables per source.

**Memory:**
- Generator: $O(d \cdot h + h^2)$
- Discriminators: $O(K \cdot (|V_k| \cdot h + h^2))$

## I. Limitations and Future Work

### Current Limitations

1. **Continuous variables only**: Current implementation assumes all variables are continuous. Categorical variables would require embedding layers.

2. **Mode collapse**: Standard GAN training instabilities apply. Wasserstein loss could improve stability.

3. **No privacy guarantees**: Synthetic records may memorize training data. Differential privacy mechanisms (DP-SGD, PATE-GAN) could be integrated.

4. **Linear projection**: Projection to source columns is linear. Nonlinear projections could capture more complex relationships.

### Planned Extensions

1. **WGAN-GP**: Replace BCE loss with Wasserstein distance and gradient penalty for training stability.

2. **Categorical support**: Add embedding layers for categorical variables with Gumbel-softmax for differentiability.

3. **Conditional generation**: Allow conditioning on policy scenarios or demographic groups.

4. **Privacy**: Integrate differential privacy via DP-SGD or teacher-student frameworks.

5. **Validation**: Add calibration to external aggregate targets (e.g., IRS SOI tables).
