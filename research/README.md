# research

The Python side of glance: the original PaddleOCR vision-pipeline prototype
(`poker_apis/`, FastAPI), its test suite and labelled datasets
(`tests/`, `test_assets/validation/`), the React dashboard prototype
(`frontend/`), notebooks (`notebooks/`) and lab setup scripts (`scripts/`).

The C++ engine in `../glance/` is the product; this folder is where ideas are
tried first. The card-vision engine (`glance/core/include/card_vision.h`) was
ported from `poker_apis/vision/`, and its accuracy gates run on
`test_assets/validation/` (see `glance-cardvision eval`).

## Docs here

- [acr.md](acr.md) — the ACR Poker scraping technique and the table-mapping
  JSON contract (kept in sync with `glance/core/include/mapping.h`).
- [slumbot.md](slumbot.md) — the Slumbot API blueprint for the control-side
  surface.

## Poker: ACR and the Slumbot control plane

Poker is glance's first target. The engine itself is poker-agnostic; this is
the domain knowledge that sits on top of it.

**ACR Poker (the table side).** The table is an [ACR](https://www.acrpoker.eu/)
window, so the "screenshot" is a grab of that window. The `acr` package
(`glance/packages/acr/`) maps the ACR 6-max table — anchor, card slots, button
areas, dealer positions, stack/bet/pot regions, named actions and the window
title. The full technique (anchor → fixed crop → per-field template match →
OCR → derived state → card vision for template-free cards) and the mapping
JSON contract are in [acr.md](acr.md).

**Slumbot (the control-plane spec).** glance's control plane mirrors the
[Slumbot API](https://slumbot.com/) — the de-facto standard poker bots are
built and benchmarked against. glance reproduces its request/response bodies,
token rules and action grammar exactly (under its own short paths, below); the
rule is *the Slumbot protocol is the standard, anything glance adds is an
extra, never a deviation.* So a bot
written for slumbot.com is a drop-in client for a glance table, and glance
clients can play against Slumbot itself unchanged.

- **Endpoints** (HTTPS POST, JSON): `POST /login`
  (`{username,password}` → `{token,...}`), `POST /new_hand`
  (`{token}`; the token may be omitted only on the very first call),
  `POST /act` (`{token, incr}`).
- **Game constants** (Slumbot house rules): blinds 50/100, stacks 200 BB,
  reset every hand; a session starts in the big blind, so play an even number
  of hands to split the button.
- **Tokens**: every request carries a `token` (except the first-ever
  `new_hand`); a response may return a *new* token after a long pause — read it
  from the previous response, never cache across hands; errors arrive as an
  `error_msg` key inside a 200 body.
- **Action strings**: a response `action` is the full hand history (streets
  separated by `/`); a request `incr` is the incremental action now. Tokens:
  `k` check, `c` call, `f` fold, `b<n>` bet/raise **to** `<n>` total for that
  street (never the delta), `/` street transition. Example `b200c/kk/kk/kb200`.
  `client_pos`: `0` = big blind, `1` = small blind.
- **Parser**: Slumbot's `sample_api.py` `ParseAction` is the ground truth
  (street, bet-to, last bettor, position, and every error path — illegal
  check/call/fold, bets too small/large, overrunning streets, `b20000c///`
  all-in, optional trailing `/`); glance ports it verbatim and unit-tests it
  against the reference. What glance adds (additive only): real-window mode
  (an ACR window instead of a benchmark server, same `incr` vocabulary driving
  clicks), the mapping between the parsed state JSON and the action grammar,
  and a state-diff event log. The glance-specific helpers (`POST /capture`,
  `GET /state`) sit outside the Slumbot contract — extras a standard client
  ignores.

The full contract, including every response shape and error path, is in
[slumbot.md](slumbot.md).

## References

- [Slumbot](https://slumbot.com/) — the bot API standard glance's control
  plane mirrors; `sample_api.py` is the reference action-string parser
- [dickreuter/Poker](https://github.com/dickreuter/Poker/) — screen-scraping technique reference
- [Gongsta/Poker-AI](https://github.com/Gongsta/Poker-AI/) — CFR + equity-based strategy reference (the consumer side of glance's parsed state)
- [ACR Poker](https://www.acrpoker.eu/)
- [Kronos](https://github.com/shiyu-coder/Kronos) — decoder-only foundation
  model for financial K-lines; tokenizer/architecture ideas for glance's own
  models (see `docs/models.md`)
- [Mamba](https://arxiv.org/abs/2312.00752) / state-space models — linear-time,
  constant-memory sequence backbone for the long screen→action session (vs
  quadratic attention); see the SSM section in `docs/models.md`
