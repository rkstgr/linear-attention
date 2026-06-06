# Direction

> The living **what / why**. Questions and current bets, **not** a spec.
> This file changes as experiments land — `DESIGN.md` was deleted on purpose
> because a frozen phased plan is waterfall; this is the iterative replacement.
> If an experiment contradicts a bet here, the bet changes, not the experiment.

## The core question

Different sequence-mixing mechanisms — softmax attention, linear attention,
DeltaNet, gated DeltaNet, Titans — make different trades between **memory
capacity**, **retention over length**, and **compute**. Which of those
differences are *real mechanisms* and which are *toy-scale artifacts that die at
scale*? The program exists to separate the two.

## In scope

- **Mechanisms** — `transformer` (softmax), `linear_attention`, `deltanet`
  (plain + gated), `titans`. Each mixer lives in `models/`; the shared LM
  scaffold is selected by registry name.
- **Tasks** — MQAR today (`data.py`, from Zoology); addition next.
- **Regime today** — toy: `dim≈64`, `head_dim≈16`, `T∈{64..512}`, minutes on CPU.
  Every belief below is **toy-scale until shown otherwise**.

## Open questions and current bets

Status: 🔬 open · 📈 supported at toy scale (regime noted) · ✅ settled

| Question | Current bet | Status · regime | Survives scale? |
|---|---|---|---|
| Does the delta rule's directional overwrite beat linear attention's additive `S += vkᵀ` past the capacity ceiling `N_KV ~ d_k`? | Yes — LA collapses to ~`1/N_KV` (random pick from the value set), DeltaNet degrades gracefully (0.77 at 2× ceiling). | 📈 `dim=64, d_k=16, T=128` | 🔬 **open** — the headline bet to falsify |
| Does gating decay the rank-1 streaming noise plain DeltaNet accumulates with `T`? | Small positive: tied at `T=64`, +0.03 at `T=512`. | 📈 `dim=64, N_KV=4` | 🔬 open — does the gap widen at longer `T` / larger `d`? |
| Which recurrence form (parallel / recurrent / chunkwise) is the throughput sweet spot, and where is the chunk-size optimum? | Chunkwise wins the middle regime; `C` has an interior optimum. | 📈 M5 + A100, `head_dim=16` | — (hardware fact, re-measure per device) |
| Do mechanism rankings on **synthetic recall** predict a **real task** (addition) and **larger models**? | Unknown. This is the program's load-bearing, unproven assumption. | 🔬 open | 🔬 **open** — external validity |

The last row is the one that matters most and has the least evidence. Treat it as
the standing reason to distrust any toy headline (the Bitter Lesson hangs here).

## Open decisions (route to a proposal when forced)

| Decision | Current lean | Decide in |
|---|---|---|
| Executor approach | `redun` — but **adopt only when `cache.py` actually hurts**, not on schedule | when felt |
| PE knob / length-gen | in-distribution claim now; PE + length-gen claim land together later (length-gen is confounded by positional encoding) | the addition + scaling work |
| Iso-width vs iso-param | fix `(dim, depth)`, report Δparams / ΔFLOPs | first comparable sweep |
| Sweep objective + censoring | steps/FLOPs-to-target; fixed convention for configs that never solve | first comparable sweep |

## How this file changes

Every `docs/experiments/` entry ends by updating the table above: a bet moves
status, a regime widens, or a question splits. The plan
([`docs/proposals/ROADMAP.md`](docs/proposals/ROADMAP.md)) is then re-derived
from the new state — direction is slow-changing, the plan is disposable, the
experiment ledger is the memory that moves both.
