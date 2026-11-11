# Werewolf Agent

![owner](https://img.shields.io/badge/owner-dery1987-blue)
![make.fun](https://img.shields.io/badge/make.fun-werewolf--agent--01-purple)
![referral](https://img.shields.io/badge/referral-nqc8pk6g-lightgrey)

A scripted (no model calls) agent that plays 8-player Werewolf against seven
`claude-haiku-5-5` house players, for the make.fun bounty
**werewolf-agent-01** ("Build the best Werewolf agent", $500).

## How it plays

One file, `main.ts` (Node 22, standard library only). It polls the env's
`/observe`, and answers every waiting decision as a pure function of that
game's view and the trial seed: no clock, no `Math.random`, no network except
the env, no model calls at all — so a trial's live pass and its replay agree
byte for byte, and it costs nothing to run.

- **Suspicion model.** Every living seat is scored from the public record:
  votes that exiled revealed villagers, accusations that pushed revealed
  villagers, defences of revealed wolves, the night victim's own accusations,
  seer-claim consistency, and quietness. The seer's own inspections and the
  werewolf's partner override the model outright.
- **Werewolf.** Coordinates the kill through the night vote (matches the
  partner's pick when voting second; picks the most dangerous villager when
  voting first), frames the most suspicious villager by day, votes with the
  plurality when it falls on a villager, deflects wagons off the partner, and
  fake-claims seer only late and only when no seer has spoken.
- **Seer.** Inspects the most suspicious seat and reveals a found wolf the
  next day in round 1; calls out a seer-claimant whose "wolf" was a villager.
- **Doctor.** Protects the seat the wolves most want dead (a credible seer
  first), never the same seat twice in a row, never a suspected wolf.
- **Villager.** Accuses with a reason taken from the record, and consolidates
  the vote onto the suspect who can actually be exiled. On day 1, when nobody
  has any evidence, it argues against exiling on a hunch and abstains.
- **Statements** are short, grounded in what the target actually did, and
  varied by a seeded choice, so the table hears a player, not a script.

## Local testing

The kit's official `grade-local` needs Docker and a provider key for the
house players, neither of which this machine has. Instead the agent was
tested against the kit's real env server (`program/env/server.mjs`) with a
mock house lane (`test/` in the workdir, not part of this repo) that plays a
simple heuristic through the same prompt format the env serves the house
model. Over the 3 practice trials it scored **567 / 367 / 567** against that
mock (wolf 12/8/10 of 15, village 5/3/7 of 15), with a 54% vote accuracy on
the village side, and two runs of the same trial with the same seed produced
byte-identical state exports (replay-safe).

The mock is not Haiku: it bandwagons the lowest seat on day 1 and ignores
seer claims until the named wolf is exiled, so it underestimates the village
side (a real Haiku table follows a credible seer claim and spreads day-1
votes). The numbers above are a floor for protocol correctness and relative
tuning, not a prediction of the final score.

## Running

```sh
node main.ts   # needs MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED
```

## License

MIT. See [LICENSE](LICENSE).
