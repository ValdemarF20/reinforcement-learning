# Exam Solutions

Self-contained implementations of all programming questions from the three previous exams.

| File | Exam | Questions |
|------|------|-----------|
| `exam2023_solutions.py` | 2023 Spring | Q13 Flower-store DP · Q14 Pendulum LQR · Q15 Bandit |
| `exam2024_solutions.py` | 2024 Spring | Q13 Bridal-gown DP · Q14 Bill's MDP · Q15 RK4 control |
| `exam2025_solutions.py` | 2025 Spring | Q13 Pendulum Euler · Q14 Cake-store DP · Q15 Q-learning |

## How to run

From the repo root (where `irlc` is importable):

```bash
python -m irlc.exam.exam_solutions.exam2023_solutions
python -m irlc.exam.exam_solutions.exam2024_solutions
python -m irlc.exam.exam_solutions.exam2025_solutions
```

Each file prints expected vs actual values for every sub-question.

## Structure

Each file follows the same pattern:
1. A model class extending the base `InventoryDPModel` / `MDP` / `ControlModel`
2. One function per sub-question (`a_...`, `b_...`, `c_...`)
3. A `__main__` block that prints sanity-check values

## Key patterns by topic

| Topic | Pattern |
|-------|---------|
| **Inventory DP** | Override `g`, `Pw`, `A`, `gN`; call `DP_stochastic` |
| **MDP** | Implement `Psr`, `A`, `is_terminal`; call `policy_evaluation` or `value_iteration` |
| **LQR** | Get `(A, B, d)` from linearization; call `LQR([A]*N, [B]*N, [d]*N, Q, R)` |
| **Q-learning** | Dict `{(s,a): float}`; update with `Q[s,a] += alpha*(r + gamma*max_Q_next - Q[s,a])` |
| **Euler** | `x = x + delta * f(x, u)` in a loop |
