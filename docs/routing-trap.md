# The routing trap, and why request-level routing escapes it

An independent September 2026 synthesis of TypeSafe founder Diogo Almeida's
design notes ("Jev Engineering for Coding Agents", 12 pp, study document, not
affiliated with TypeSafe) carries the cleanest statement of a real cost trap
in model routing, and every routing library -- this one included -- has to
answer it. This is our answer.

## The trap, in their arithmetic

Delegating execution to a cheaper model costs a context-load on the way down
(the cheap model must load the session) and a reprocessing pass on the way
back (the frontier model must re-read what changed). Plugging in a plausible
agent-session shape (65% context, 12% generation, 23% reading), pure frontier
costs **4.15** and the routed path costs **6.19**: the route that was supposed
to save money costs half again as much. Routing priced per token is wrong;
routing must be priced **per context rebuild**.

## Why request-level routing does not fall in

The trap needs a shared, expensive context that has to be reprocessed per
delegation. jev-route has none of that by design:

- each routing decision is **one small bounded excerpt** -- a few hundred
  tokens, head+tail of the request, never the session transcript;
- the excerpt is identical whichever tier the decision goes to, so choosing
  cheap over frontier costs **zero context rebuild**;
- the decision model is the *small* model by construction (Jev 421M-class, or
  the local Laya checkpoint), paid once per request, at a measured fraction of
  the routed model's price.

Measured on our own backtest (`docs/backtest_v0.2.md`): the decision call adds
well under 2% of the savings it buys on the 500-request trace. The trap's
context-rebuild term is zero for us, and the doc's own escape hatch -- "hand
the cheaper model a small, purpose-built context instead of the full
transcript" -- is literally what the excerpt already is.

## Where the trap *does* bite, and the honest answer

The trap applies to **agent-session** integrations -- a router embedded in a
long-lived coding agent where "route to cheap" means handing the session to a
second model. Two of our mechanisms exist precisely for that case:

- **the small-context principle is load-bearing**: integrations should send the
  excerpt, never the transcript (the LiteLLM hooks enforce exactly this);
- **cross-provider quota tandem** (v0.5) flips to the *same model* on another
  provider's quota rather than degrading to a different model, so a capacity
  flip never changes the context the request was priced for.

Source: "Jev Engineering for Coding Agents", Sept 2026, independent synthesis
based on design notes by Diogo Almeida (TypeSafe). Not affiliated with or
endorsed by TypeSafe; the arithmetic and figures are as printed there.
