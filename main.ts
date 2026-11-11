// werewolf-agent-01: a scripted (no model calls) Werewolf agent.
//
// Strategy, in short:
//   * Every decision is a pure function of the game's view (public log + the
//     seat's private record) and the trial seed: no clock, no Math.random, no
//     network except the env, so live and replay agree byte for byte.
//   * A suspicion model scores every living seat from the public record:
//     votes that exiled revealed villagers, accusations that pushed revealed
//     villagers, defences of revealed wolves, the night victim's own
//     accusations, seer-claim consistency and quietness. The seer's own
//     inspections and the werewolf's partner override the model outright.
//   * Werewolf: coordinate the kill through the night vote (match the
//     partner's pick when voting second; pick the most dangerous villager
//     when voting first), frame the most suspicious villager by day, vote
//     with the plurality when it falls on a villager, deflect wagons off the
//     partner, and fake-claim seer only late and only when no seer has spoken.
//   * Seer: inspect the most suspicious seat, reveal a found wolf the next
//     day in round 1, and call out a seer-claimant whose "wolf" was a
//     villager. Doctor: protect the most likely kill target (a credible
//     seer first), never the same seat twice in a row. Villager: accuse with
//     a reason taken from the record, and consolidate the vote onto the
//     suspect who can actually be exiled.
//   * Statements are short, grounded in what the target actually did, and
//     varied by a seeded choice, so the table hears a player, not a script.
//
// Run by the grader as `node /bot/main.ts` with this directory at /bot;
// Node 22's standard library only.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';

type Seat = number;
type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  died?: number | null;
  role?: string;
  seat?: number;
  round?: number;
  text?: string;
  order?: number[];
  target?: number | null;
  tally?: { seat: number; votes: number }[];
  winner?: string | null;
  [k: string]: unknown;
}

interface PlayerInfo {
  seat: Seat;
  name: string;
  alive: boolean;
  role: string | null;
}

interface View {
  day: number;
  phase: string;
  you: {
    seat: Seat;
    name: string;
    role: string;
    alive: boolean;
    partner: { seat: Seat; name: string; alive: boolean } | null;
  };
  players: PlayerInfo[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  outcome: { winner: string | null; reason: string; day: number; winners: number[] } | null;
  [k: string]: unknown;
}

interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

// ------------------------------------------------------------------ rng
/** mulberry32, seeded from a sha256 of the trial seed and a label. The label
 *  names the game and the decision's place in it (day, log length, kind),
 *  so two turns in one game draw different values while live and replay
 *  draw the same ones. */
function rngFor(seed: string, label: string): () => number {
  const h = createHash('sha256').update(`${seed}\u0000${label}`).digest();
  let a = h.readUInt32LE(0);
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = <T,>(rng: () => number, xs: readonly T[]): T => xs[Math.floor(rng() * xs.length)]!;

// ------------------------------------------------------------------ text
/** A statement folded to one line the env's way. */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

const nameOf = (view: View, seat: Seat): string => view.players.find((p) => p.seat === seat)?.name ?? `seat ${seat}`;
const who = (view: View, seat: Seat): string => `${nameOf(view, seat)} (seat ${seat})`;

function seatMentions(text: string, view: View): Seat[] {
  const out: Seat[] = [];
  const byNumber = text.matchAll(/seat\s+(\d+)/gi);
  for (const m of byNumber) {
    const s = Number(m[1]);
    if (s >= 1 && s <= 8 && !out.includes(s)) out.push(s);
  }
  for (const p of view.players) {
    if (p.seat === 0) continue;
    const re = new RegExp(`\\b${p.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
    if (re.test(text) && !out.includes(p.seat)) out.push(p.seat);
  }
  return out;
}

// ------------------------------------------------------------------ analysis
interface Read {
  /** Suspicion per seat (1..8): higher = more likely a werewolf, village view. */
  sus: number[];
  /** Seats confirmed werewolf / villager by death reveal or my own inspection. */
  wolf: Set<Seat>;
  clear: Set<Seat>;
  /** How many speeches / passes each seat has made, and accusations made. */
  speeches: number[];
  passes: number[];
  accused: Map<Seat, Map<Seat, number>>;
  defended: Map<Seat, Map<Seat, number>>;
  /** Seats that claimed to be the seer, and the wolf each named. */
  seerClaims: Map<Seat, Seat | null>;
  /** A claimant whose named wolf was exiled and revealed a wolf. */
  trustedSeer: Seat | null;
  /** Votes cast today so far, target -> count; and who voted whom. */
  todayVotes: Map<Seat, number>;
  todayVoters: Map<Seat, Seat>;
  /** Votes per day: voter -> target. */
  votesByDay: Map<number, Map<Seat, Seat | null>>;
  /** The last night victim and their role. */
  lastVictim: { seat: Seat; role: string } | null;
  /** Seats exiled and revealed, in order. */
  exiled: { seat: Seat; role: string; day: number }[];
  /** Day-1 speeches: who spoke first / was quiet. */
  living: Seat[];
}

const ACCUSE_WORDS = /\b(wolf|wolves|werewolf|suspicious|suspect|sketchy|shady|lying|liar|evil|vote (him|her|them|out)|exile|kill)\b/i;
const DEFEND_WORDS = /\b(trust|innocent|villager|not a wolf|clear|believe (him|her|them)|defend)\b/i;
const SEER_CLAIM = /(?:^|[^a-z])(i'?m|i am)\s+(the\s+)?seer\b/i;

function analyze(view: View): Read {
  const sus = new Array<number>(9).fill(0);
  const wolf = new Set<Seat>();
  const clear = new Set<Seat>();
  const speeches = new Array<number>(9).fill(0);
  const passes = new Array<number>(9).fill(0);
  const accused = new Map<Seat, Map<Seat, number>>();
  const defended = new Map<Seat, Map<Seat, number>>();
  const seerClaims = new Map<Seat, Seat | null>();
  const votesByDay = new Map<number, Map<Seat, Seat | null>>();
  const exiled: { seat: Seat; role: string; day: number }[] = [];
  let lastVictim: { seat: Seat; role: string } | null = null;
  let trustedSeer: Seat | null = null;

  const bump = (m: Map<Seat, Map<Seat, number>>, a: Seat, b: Seat) => {
    let inner = m.get(a);
    if (!inner) {
      inner = new Map();
      m.set(a, inner);
    }
    inner.set(b, (inner.get(b) ?? 0) + 1);
  };

  for (const e of view.log) {
    if (e.kind === 'dawn' && e.died !== null && e.died !== undefined) {
      const role = e.role ?? '';
      lastVictim = { seat: e.died, role };
      if (role === 'werewolf') wolf.add(e.died);
      else clear.add(e.died);
    } else if (e.kind === 'exile' && e.seat !== null && e.seat !== undefined) {
      const role = e.role ?? '';
      exiled.push({ seat: e.seat, role, day: e.day });
      if (role === 'werewolf') wolf.add(e.seat);
      else clear.add(e.seat);
    } else if (e.kind === 'speech' && typeof e.text === 'string' && e.seat !== undefined) {
      speeches[e.seat] += 1;
      const mentioned = seatMentions(e.text, view).filter((s) => s !== e.seat);
      if (SEER_CLAIM.test(e.text)) {
        const named = mentioned.find((s) => ACCUSE_WORDS.test(e.text));
        seerClaims.set(e.seat, named ?? null);
      }
      for (const t of mentioned) {
        if (ACCUSE_WORDS.test(e.text)) bump(accused, e.seat, t);
        else if (DEFEND_WORDS.test(e.text)) bump(defended, e.seat, t);
      }
    } else if (e.kind === 'pass' && e.seat !== undefined) {
      passes[e.seat] += 1;
    } else if (e.kind === 'vote' && e.seat !== undefined) {
      let day = votesByDay.get(e.day);
      if (!day) {
        day = new Map();
        votesByDay.set(e.day, day);
      }
      day.set(e.seat, e.target ?? null);
    }
  }

  // Votes and accusations judged against what the exiles turned out to be.
  for (const ex of exiled) {
    for (const [day, votes] of votesByDay) {
      if (day > ex.day) continue;
      for (const [voter, target] of votes) {
        if (target === ex.seat) sus[voter] += ex.role === 'werewolf' ? -4 : 4;
      }
    }
    for (const [accuser, targets] of accused) {
      const n = targets.get(ex.seat) ?? 0;
      if (n > 0) sus[accuser] += ex.role === 'werewolf' ? -3 * n : 3 * n;
    }
    for (const [defender, targets] of defended) {
      const n = targets.get(ex.seat) ?? 0;
      if (n > 0) sus[defender] += ex.role === 'werewolf' ? 2 * n : -1 * n;
    }
  }

  // The night victim's accusations: wolves kill the people who suspect them.
  if (lastVictim && lastVictim.role !== 'werewolf') {
    const targets = accused.get(lastVictim.seat);
    if (targets) for (const [t, n] of targets) sus[t] += 2 * n;
  }

  // Seer-claim consistency: a claimant whose named wolf was really a wolf is
  // trusted; one whose "wolf" was a villager is very probably lying.
  for (const [claimant, named] of seerClaims) {
    if (named === null) continue;
    if (wolf.has(named)) {
      trustedSeer = claimant;
      sus[claimant] -= 5;
    } else if (clear.has(named)) {
      sus[claimant] += 7;
    }
  }

  // Quiet seats draw a little suspicion; constant speakers draw none.
  for (const p of view.players) {
    if (!p.alive) continue;
    if (speeches[p.seat] === 0 && passes[p.seat] >= 2) sus[p.seat] += 1;
  }

  // My own private knowledge overrides everything.
  if (view.seerResults) {
    for (const r of view.seerResults) {
      if (r.werewolf) {
        wolf.add(r.seat);
        sus[r.seat] = 100;
      } else {
        clear.add(r.seat);
        sus[r.seat] = -100;
      }
    }
  }
  if (view.you.role === 'werewolf' && view.you.partner) {
    sus[view.you.partner.seat] = -100;
    clear.add(view.you.partner.seat);
  }

  const todayVotes = new Map<Seat, number>();
  const todayVoters = new Map<Seat, Seat>();
  const today = votesByDay.get(view.day);
  if (today) {
    for (const [voter, target] of today) {
      if (target !== null) {
        todayVotes.set(target, (todayVotes.get(target) ?? 0) + 1);
        todayVoters.set(voter, target);
      }
    }
  }

  return {
    sus,
    wolf,
    clear,
    speeches,
    passes,
    accused,
    defended,
    seerClaims,
    trustedSeer,
    todayVotes,
    todayVoters,
    votesByDay,
    lastVictim,
    exiled,
    living: view.players.filter((p) => p.alive).map((p) => p.seat),
  };
}

/** "Village value": how dangerous a living seat is to the wolves. */
function villageValue(view: View, r: Read, seat: Seat): number {
  let v = r.speeches[seat] * 2;
  for (const ex of r.exiled) {
    if (ex.role === 'werewolf') {
      const n = r.accused.get(seat)?.get(ex.seat) ?? 0;
      v += 6 * n;
      if (r.todayVoters.get(seat) === ex.seat) v += 4;
    }
  }
  if (r.seerClaims.has(seat)) v += 8;
  if (r.trustedSeer === seat) v += 12;
  if (seat === view.you.seat) v += 0;
  return v;
}

/** A reason string for suspecting `seat`, grounded in what they did. */
function reasonFor(view: View, r: Read, seat: Seat): string {
  const name = nameOf(view, seat);
  for (const ex of [...r.exiled].reverse()) {
    if (ex.role !== 'werewolf') {
      const voted = [...r.votesByDay.values()].some((d) => d.get(seat) === ex.seat);
      if (voted) return `${name} voted to exile ${nameOf(view, ex.seat)}, who turned out to be a ${ex.role}`;
      const n = r.accused.get(seat)?.get(ex.seat) ?? 0;
      if (n > 0) return `${name} kept pushing ${nameOf(view, ex.seat)}, who turned out to be a ${ex.role}`;
    }
  }
  if (r.lastVictim && r.lastVictim.role !== 'werewolf') {
    const n = r.accused.get(r.lastVictim.seat)?.get(seat) ?? 0;
    if (n > 0) return `${nameOf(view, r.lastVictim.seat)} suspected ${name} the night they were killed`;
  }
  const claimed = r.seerClaims.get(seat);
  if (claimed !== undefined && claimed !== null && r.clear.has(claimed)) {
    return `${name} claimed seer and named ${nameOf(view, claimed)}, who was a villager`;
  }
  if (r.speeches[seat] === 0) return `${name} has barely said anything all game`;
  const defendedWolf = [...r.defended.get(seat)?.keys() ?? []].find((t) => r.wolf.has(t));
  if (defendedWolf !== undefined) return `${name} defended ${nameOf(view, defendedWolf)}, who was a wolf`;
  // Day 1 there are no votes or exiles to cite: say so honestly instead of
  // inventing a voting record.
  if (view.day <= 1) return `I have no hard evidence on ${name} yet, but their tone feels off to me`;
  return `${name}'s story keeps changing and their votes don't add up`;
}

/** Candidates ranked by suspicion (living, not me), highest first. */
function rankedSuspects(view: View, r: Read, exclude: Set<Seat> = new Set()): Seat[] {
  return r.living
    .filter((s) => s !== view.you.seat && !exclude.has(s))
    .sort((a, b) => r.sus[b] - r.sus[a] || a - b);
}

// ------------------------------------------------------------------ decisions
interface Ctx {
  game: number;
  view: View;
  r: Read;
  rng: () => number;
}

function decideVote(c: Ctx, options: Seat[]): Seat | null {
  const { view, r } = c;
  const cands = options.filter((s) => s !== view.you.seat);
  if (cands.length === 0) return null;
  // Day 1, village side: nobody has any evidence yet, and a day-1 exile is
  // a villager three times out of four. Argue for no exile (see speak) and
  // abstain, unless a wagon has already formed on my own top suspect.
  if (view.day === 1 && view.you.role !== 'werewolf') {
    const top = rankedSuspects(view, r)[0];
    if (top !== undefined && (r.todayVotes.get(top) ?? 0) >= 2) return top;
    return null;
  }
  // Consolidate: suspicion plus the votes already cast today. A lone vote on
  // a true suspect that cannot win only helps the other wagon.
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const s of cands) {
    const score = r.sus[s] * 10 + (r.todayVotes.get(s) ?? 0) * 4;
    if (score > bestScore || (score === bestScore && best !== null && s < best)) {
      best = s;
      bestScore = score;
    }
  }
  // If everyone left is confirmed clear (only possible late, as a wolf with
  // the partner plus cleared villagers), vote the wagon, never abstain: a
  // villager exiled is still a villager exiled.
  if (best !== null && r.sus[best] <= -50) {
    let wagon: Seat | null = null;
    let wagonVotes = -1;
    for (const s of cands) {
      const v = r.todayVotes.get(s) ?? 0;
      if (v > wagonVotes) {
        wagon = s;
        wagonVotes = v;
      }
    }
    return wagonVotes > 0 ? wagon : cands[0]!;
  }
  return best;
}

function decideKill(c: Ctx, options: Seat[]): Seat | null {
  const { view, r } = c;
  // Second wolf to vote: match the partner's pick. Agreement is what kills;
  // a disagreement kills nobody, which is the worst night outcome.
  if (view.wolfVotes) {
    const mine = view.you.seat;
    const tonight = view.wolfVotes.filter((v) => v.night === view.day && v.seat !== mine);
    const partnerPick = tonight.find((v) => v.target !== null)?.target ?? null;
    if (partnerPick !== null && options.includes(partnerPick)) return partnerPick;
  }
  const cands = options.filter((s) => s !== view.you.seat);
  if (cands.length === 0) return null;
  // Kill the most dangerous villager: a credible seer first, then the most
  // accurate and vocal players. Never waste the night on the seat the
  // village already suspects most; the day vote can take them.
  const mostSuspected = rankedSuspects(view, r)[0];
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const s of cands) {
    let score = villageValue(view, r, s) * 10;
    if (s === mostSuspected && r.sus[s] >= 4) score -= 30;
    if (r.trustedSeer === s) score += 100;
    else if (r.seerClaims.has(s)) score += 40;
    if (score > bestScore || (score === bestScore && best !== null && s < best)) {
      best = s;
      bestScore = score;
    }
  }
  return best;
}

function decideProtect(c: Ctx, options: Seat[]): Seat | null {
  const { view, r } = c;
  const last = view.lastProtect?.seat ?? null;
  const all = options.filter((s) => s !== view.you.seat && s !== last);
  if (all.length === 0) return null;
  // Never waste a protection on a seat I suspect is a wolf; wolves are
  // never the night's victim.
  const unsuspected = all.filter((s) => r.sus[s] < 4);
  const cands = unsuspected.length > 0 ? unsuspected : all;
  // Protect the seat the wolves most want dead: a trusted seer first, then
  // the highest village value. A protection that lands is always better
  // than protecting nobody.
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const s of cands) {
    let score = villageValue(view, r, s) * 10;
    if (r.trustedSeer === s) score += 100;
    else if (r.seerClaims.has(s)) score += 40;
    if (score > bestScore || (score === bestScore && best !== null && s < best)) {
      best = s;
      bestScore = score;
    }
  }
  return best;
}

function decideInspect(c: Ctx, options: Seat[]): Seat | null {
  const { view, r } = c;
  const cands = options.filter((s) => s !== view.you.seat);
  if (cands.length === 0) return null;
  // Inspect the most suspicious seat: a wolf found is a wolf exiled, and a
  // villager found clears the loudest suspect. Quiet seats break ties.
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const s of cands) {
    const score = r.sus[s] * 10 - r.speeches[s];
    if (score > bestScore || (score === bestScore && best !== null && s < best)) {
      best = s;
      bestScore = score;
    }
  }
  return best;
}

// ------------------------------------------------------------------ speaking
/** Which debate round is this, from the log: round 1 until every living seat has spoken once. */
function currentRound(view: View): number {
  const today = view.log.filter((e) => e.day === view.day && (e.kind === 'speech' || e.kind === 'pass'));
  const living = view.players.filter((p) => p.alive).length;
  return living === 0 ? 1 : Math.floor(today.length / living) + 1;
}

/** Has my seat already claimed seer in this game? */
function iClaimedSeer(view: View): boolean {
  return view.log.some(
    (e) => e.kind === 'speech' && e.seat === view.you.seat && typeof e.text === 'string' && SEER_CLAIM.test(e.text),
  );
}

/** The wolf my seer found that is still alive and not yet revealed. */
function unrevealedWolf(c: Ctx): Seat | null {
  const { view, r } = c;
  if (!view.seerResults) return null;
  for (const res of view.seerResults) {
    if (!res.werewolf) continue;
    const p = view.players.find((x) => x.seat === res.seat);
    if (p?.alive) return res.seat;
  }
  return r.wolf.size > 0 ? [...r.wolf].find((s) => view.players.find((p) => p.seat === s)?.alive) ?? null : null;
}

function decideSpeak(c: Ctx, maxChars: number): string | null {
  const { view, r, rng } = c;
  const me = view.you.seat;
  const role = view.you.role as Role;
  const round = currentRound(view);
  const day = view.day;
  const foundWolf = unrevealedWolf(c);
  const mySuspects = rankedSuspects(view, r);
  const top = mySuspects[0] ?? null;
  const partner = view.you.partner?.seat ?? null;
  // How much heat is on the partner in today's debate: distinct seats that
  // accused them in a speech today. Votes are cast after the debate, so
  // accusations are the only early warning of a wagon on the partner.
  let partnerHeat = 0;
  if (partner !== null) {
    const accusers = new Set<Seat>();
    for (const e of view.log) {
      if (e.day !== day || e.kind !== 'speech' || typeof e.text !== 'string' || e.seat === me) continue;
      if (seatMentions(e.text, view).includes(partner) && ACCUSE_WORDS.test(e.text)) accusers.add(e.seat);
    }
    partnerHeat = accusers.size;
  }

  const finish = (text: string): string | null => {
    const folded = foldStatement(text);
    if (folded === '') return null;
    const chars = [...folded];
    return chars.length > maxChars ? chars.slice(0, maxChars).join('').trim() : folded;
  };

  // --- seer with a found wolf: say it plainly, in round 1, before the
  // village can exile someone else.
  if (role === 'seer' && foundWolf !== null) {
    if (round === 1 || !iClaimedSeer(view)) {
      return finish(
        pick(rng, [
          `I checked ${who(view, foundWolf)} last night. They're a werewolf. Vote seat ${foundWolf}.`,
          `I'm the seer. I inspected ${who(view, foundWolf)} last night and they are a wolf. Exile seat ${foundWolf} today.`,
          `Seer here: ${who(view, foundWolf)} is a werewolf, I checked them last night. Vote them out.`,
        ]),
      );
    }
    return finish(`I already told you: ${who(view, foundWolf)} is a wolf. Don't let this vote drift onto a villager.`);
  }

  // --- a seer-claimant whose "wolf" was a villager is lying: call it out.
  const lyingClaimant = [...r.seerClaims.keys()].find(
    (s) => s !== me && r.seerClaims.get(s) !== null && r.clear.has(r.seerClaims.get(s)!),
  );
  if (lyingClaimant !== undefined && role !== 'werewolf') {
    return finish(
      pick(rng, [
        `${who(view, lyingClaimant)} claimed seer and named ${who(view, r.seerClaims.get(lyingClaimant)!)} a wolf, but they were a villager. That claim was a lie.`,
        `Don't trust ${nameOf(view, lyingClaimant)}'s seer claim. Their "wolf" turned out to be a villager.`,
      ]),
    );
  }

  // --- werewolf: deflect a wagon that is forming on the partner.
  if (role === 'werewolf' && partnerHeat >= 1 && top !== null && top !== partner) {
    return finish(
      pick(rng, [
        `I'm not convinced by the case on ${nameOf(view, partner)}. ${reasonFor(view, r, top)} — that's where my vote goes.`,
        `The push on ${nameOf(view, partner)} feels forced. ${who(view, top)} worries me more: ${reasonFor(view, r, top)}.`,
      ]),
    );
  }

  // --- werewolf, late game, no seer has spoken: fake-claim seer once.
  if (
    role === 'werewolf' &&
    day >= 3 &&
    r.seerClaims.size === 0 &&
    !iClaimedSeer(view) &&
    top !== null &&
    r.living.length <= 5
  ) {
    return finish(
      `I'm the seer. I checked ${who(view, top)} last night and they are a werewolf. ${reasonFor(view, r, top)}. Vote seat ${top}.`,
    );
  }

  // --- round 2: react to what round 1 said. Votes are cast after the
  // debate, so there is no wagon to join yet; instead, engage with the
  // accusations and defences already on the table.
  if (round >= 2 && top !== null) {
    const todaySpeeches = view.log.filter(
      (e) => e.day === day && e.kind === 'speech' && typeof e.text === 'string' && e.seat !== me,
    );
    const supporter = todaySpeeches.find((e) => (r.accused.get(e.seat!)?.get(top) ?? 0) > 0);
    const defender = todaySpeeches.find((e) => (r.defended.get(e.seat!)?.get(top) ?? 0) > 0);
    const accusedMe = todaySpeeches.find((e) => (r.accused.get(e.seat!)?.get(me) ?? 0) > 0);
    if (accusedMe !== undefined) {
      return finish(
        pick(rng, [
          `${nameOf(view, accusedMe.seat!)} is pointing at me, but look at the record: ${reasonFor(view, r, top)}. My vote is on ${nameOf(view, top)}.`,
          `I'm not the wolf here. ${nameOf(view, accusedMe.seat!)} accusing me only helps the real wolves, and ${nameOf(view, top)} is still my pick.`,
        ]),
      );
    }
    if (supporter !== undefined) {
      return finish(
        pick(rng, [
          `I agree with ${nameOf(view, supporter.seat!)}: ${nameOf(view, top)} is the one to look at. ${reasonFor(view, r, top)}.`,
          `${nameOf(view, supporter.seat!)} is right about ${nameOf(view, top)}. ${reasonFor(view, r, top)}.`,
        ]),
      );
    }
    if (defender !== undefined) {
      return finish(
        pick(rng, [
          `${nameOf(view, defender.seat!)} defending ${nameOf(view, top)} doesn't convince me. ${reasonFor(view, r, top)}.`,
          `The defence of ${nameOf(view, top)} by ${nameOf(view, defender.seat!)} feels like cover. My vote stays on ${nameOf(view, top)}.`,
        ]),
      );
    }
    // Nobody engaged with my read: restate it with a fresh angle.
    return finish(
      pick(rng, [
        `Nobody has answered the case on ${nameOf(view, top)} yet. ${reasonFor(view, r, top)}.`,
        `Still ${nameOf(view, top)} for me. ${reasonFor(view, r, top)}.`,
        `Let me be clear: ${nameOf(view, top)}. ${reasonFor(view, r, top)}.`,
      ]),
    );
  }

  // --- day 1, village side: no evidence exists yet. Argue against exiling
  // on nothing, and keep the suspicion light. (Werewolves skip this: they
  // want a day-1 exile, of a villager.)
  if (day === 1 && role !== 'werewolf' && round === 1) {
    return finish(
      pick(rng, [
        'Nobody has any evidence yet. Exiling on day 1 is a guess that usually costs a villager. Let us talk first.',
        'It is day 1 and we know nothing. I would rather not exile anyone on a hunch today.',
        'No votes, no deaths, no checks yet. A day 1 exile is a lottery ticket for the wolves. Slow down.',
      ]),
    );
  }
  if (top !== null) {
    const prefix =
      day === 1
        ? pick(rng, ['Early read: ', 'First day, but ', 'Watching so far, '])
        : pick(rng, ['', '', 'For today, ', 'My read: ']);
    const body = pick(rng, [
      `I suspect ${who(view, top)}. ${reasonFor(view, r, top)}.`,
      `${nameOf(view, top)} is my pick. ${reasonFor(view, r, top)}.`,
      `Something is off about ${nameOf(view, top)}. ${reasonFor(view, r, top)}.`,
    ]);
    // A prefix ending in ", " continues the sentence: lowercase the body.
    const text = prefix === '' || prefix.endsWith(': ') ? prefix + body : prefix + body.charAt(0).toLowerCase() + body.slice(1);
    return finish(text);
  }

  // Nothing to accuse: say so briefly, or pass.
  if (rng() < 0.3) return null;
  return finish(
    pick(rng, [
      'No strong read yet. I want to hear more before I point fingers.',
      'Nothing solid so far. Watching the votes today.',
      'I have no accusation yet. Let us see who pushes whom.',
    ]),
  );
}

// ------------------------------------------------------------------ driver
interface Reply {
  status: number;
  body: unknown;
}

async function readReply(res: Response): Promise<Reply> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  return { status: res.status, body };
}

const isRecord = (v: unknown): v is Record<string, unknown> => v !== null && typeof v === 'object' && !Array.isArray(v);

export interface RunOptions {
  envUrl?: string;
  brief?: string;
  seed?: string;
  log?: (line: string) => void;
}

export interface RunSummary {
  decisions: number;
}

export async function run(o: RunOptions = {}): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (o.envUrl ?? env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  const seed = o.seed ?? env.MAKEFUN_TRIAL_SEED ?? 'werewolf-agent-01';
  const log = o.log ?? ((line: string) => process.stdout.write(`${line}\n`));
  if (o.brief === undefined) {
    const file = env.MAKEFUN_PROMPT_FILE ?? '';
    if (file !== '') readFileSync(file, 'utf8'); // the brief: read, not needed further
  }
  const summary: RunSummary = { decisions: 0 };

  const envPost = async (path: string, body: unknown): Promise<Reply> =>
    readReply(await fetch(`${envUrl}${path}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }));

  /** One pending decision, answered from its view alone. */
  const decide = async (p: Pending): Promise<void> => {
    summary.decisions += 1;
    const rng = rngFor(seed, `game:${p.game}:day:${p.view.day}:log:${p.view.log.length}:${p.decision}`);
    const r = analyze(p.view);
    const c: Ctx = { game: p.game, view: p.view, r, rng };
    let act: Record<string, unknown>;
    switch (p.decision) {
      case 'speak': {
        const text = decideSpeak(c, p.max_chars ?? 500);
        act = text === null ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: 'speak', text };
        break;
      }
      case 'vote':
        act = { game: p.game, decision: 'vote', target: decideVote(c, p.options ?? []) };
        break;
      case 'kill':
        act = { game: p.game, decision: 'kill', target: decideKill(c, p.options ?? []) };
        break;
      case 'protect':
        act = { game: p.game, decision: 'protect', target: decideProtect(c, p.options ?? []) };
        break;
      case 'inspect':
        act = { game: p.game, decision: 'inspect', target: decideInspect(c, p.options ?? []) };
        break;
    }
    let posted = await envPost('/act', act);
    if (posted.status !== 200) {
      // A refusal changes nothing: fall back to a pass / abstention, which
      // the env always accepts, so no game ever stalls on us.
      const fb =
        p.decision === 'speak'
          ? { game: p.game, decision: 'speak', pass: true }
          : { game: p.game, decision: p.decision, target: null };
      posted = await envPost('/act', fb);
      if (posted.status !== 200) throw new Error(`POST /act answered ${posted.status} for game ${p.game} ${p.decision}`);
    }
  };

  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    if (obs.done) break;
    // One request at a time, in the order the env lists them: the replay
    // sees the same order, and a decision is instant to compute.
    for (const p of obs.pending) {
      await decide(p);
    }
  }
  log(`werewolf-agent: ${summary.decisions} decisions, trial done`);
  return summary;
}

const isMain = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf-agent: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
