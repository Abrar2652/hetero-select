
---

### 1. Score Component Ablations (Deconstructing $S_k(t)$)
Your central claim is that a single composite score $S_k(t)$ can cleanly drive all four stages of a federated round [1]. You must prove the necessity of each of the four components in Equation (1) [1]:
$$S_k(t) = V'_k(t) + \lambda_D D_k(t) + \lambda_F F'_k(t) + \lambda_{St} St'_k(t)$$

You should run experiments where you turn off these components one-by-one ($\lambda_i = 0$):
*   **Without Loss Informativeness ($V'_k = 0$):** Shows what happens when selection and compression are blind to local convergence progress [1].
*   **Without Diversity ($\lambda_D = 0$):** Demonstrates how much non-IID data hurts the framework when you do not actively prioritize gradient-direction diversity [1].
*   **Without Fairness/Staleness ($\lambda_F = 0$ or $\lambda_{St} = 0$):** This is a critical stability baseline [1]. Without these, loss-greedy selection will cause the global model to overfit to a subset of high-loss clients [1]. Show how the absence of these components leads to late-stage accuracy drops or high variance.

---

### 2. The Newton-Guided Sparsifier vs. Magnitude Sparsification
One of your primary technical contributions is the **Newton-guided top-$k$ sparsifier** utilizing a diagonal Hessian estimate [1]. Reviewers will want to see if this mathematical overhead is actually worth it [1].
*   **Newton vs. Pure Magnitude:** Compare your Newton-guided Top-$k$ sparsification against standard magnitude-based Top-$k$ on all layers [1]. Show how much of your accuracy/time gain comes strictly from using the Gauss-Southwell Newton criterion [1].
*   **Sensitivity to Layer Budget ($Q$):** You set $Q = 3$ layers to balance computational overhead (Hessian-vector products) and convergence rate [1]. Show an ablation of $Q \in \{0, 1, 3, 5, \text{All}\}$ [1]. This proves that $Q = 3$ is indeed the "sweet spot" where the extra backpropagation step is minimized while retaining optimal gradient-coordinate selection [1].

---

### 3. Adaptive Error Feedback ($\beta$) vs. Static Error Feedback
In Equation (9), you introduce an adaptive schedule where the error-buffer decay rate $\beta(t)$ is tied to the global compression budget $\theta_t$ [1].
*   **Static $\beta$ Baseline:** Compare your adaptive schedule against standard, fixed error-feedback baselines (e.g., static $\beta = 0.9$ or $0.95$, as commonly used in EF21) [1]. 
*   **The "Buffer Collapse" Proof:** Demonstrate how a static error-feedback buffer suffers from residual decay collapse when the compression budget becomes extremely tight in late rounds, whereas your adaptive scheduler prolongs residual survival [1].

---

### 4. Score-Adaptive Learning Rate & Server Momentum
Your framework adaptive-scales the local learning rate $\eta_k(t)$ and applies score-weighted server aggregation [1].
*   **Uniform LR vs. Score-Adaptive LR:** Run a baseline with a standard, uniform learning rate for all clients, and compare it to your score-adaptive Equation (12) [1]. This proves that letting highly informative clients make more local progress directly aids convergence [1].
*   **Uniform Aggregation vs. Score-Weighted Aggregation:** Compare standard FedAvg aggregation (equal weights for all selected clients) against your score-weighted server aggregation (Equation 13) [1]. This highlights whether biasing the aggregate toward high-informativeness gradients prevents client drift [1].

---

### 5. Empirically Validating Theoretical Bounds
Reviewers love it when the empirical section directly validates the paper's theoretical theorems [1].
*   **Plotting Selected Heterogeneity ($B^2_{\text{sel}}$):** In Theorem IV.1, you prove that score-proportional sampling reduces the effective selected heterogeneity $B^2_{\text{sel}}$ below the uniform baseline $B^2$ [1]. 
*   **Ablation/Validation Plot:** Actually calculate and plot the value of $B^2_{\text{sel}}$ at each round of your simulation for both uniform sampling and HeteRo-Select. Showing a line graph where HeteRo-Select’s $B^2_{\text{sel}}$ stays consistently lower than the uniform baseline provides immediate, visual proof of Theorem IV.1's validity [1].

---

### Suggested Structure for the Ablation Table in Your Draft

To present these cleanly, you can structure a consolidated ablation table like this:

| Ablation Category | Variant | CIFAR-10 Peak Acc (%) | Time-to-Target (s) | Traffic-to-Target (MB) |
| :--- | :--- | :---: | :---: | :---: |
| **Primary** | **HeteRo-Select (Primary)** [1] | **73.03%** | **2,913** | **2,010** |
| *1. Score Components* | w/o Loss ($V'_k = 0$) [1] | *[Value]* | *[Value]* | *[Value]* |
| | w/o Diversity ($\lambda_D = 0$) [1] | *[Value]* | *[Value]* | *[Value]* |
| | w/o Fairness & Staleness ($\lambda_F, \lambda_{St} = 0$) [1] | *[Value]* | *[Value]* | *[Value]* |
| *2. Sparsifier* | Pure Magnitude Top-$k$ ($Q=0$) [1] | *[Value]* | *[Value]* | *[Value]* |
| | Full-Model Newton ($Q=\text{All}$) [1] | *[Value]* | *[Value]* | *[Value]* |
| *3. Error Feedback* | Static Error Feedback ($\beta=0.9$) [1] | *[Value]* | *[Value]* | *[Value]* |
| *4. Learning Rate* | Uniform Learning Rate ($\eta_k = \eta_0$) [1] | *[Value]* | *[Value]* | *[Value]* |
| *5. Aggregation* | Uniform Aggregation (FedAvg) [1] | *[Value]* | *[Value]* | *[Value]* |

By presenting this deep level of decomposition, you head off the common reviewer criticism of: *"How do we know which part of HeteRo-Select is actually yielding the performance gains?"*