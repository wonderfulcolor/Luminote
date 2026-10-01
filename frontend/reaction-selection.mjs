// Dominance separates feelings that share valence and arousal (anger vs. fear), but it is
// less reliable than the other two axes, so it counts at half weight by default.
export const DEFAULT_DOMINANCE_WEIGHT = 0.5;
// Recently shown reactions are nudged back so the same image does not repeat every turn.
export const DEFAULT_RECENCY_PENALTY = 0.15;

function reactionDistance(asset, point, dominanceWeight) {
  const planar = (point.valence - asset.valence) ** 2 + (point.arousal - asset.arousal) ** 2;
  // Assets without a dominance coordinate are matched on valence and arousal alone.
  const dominance = Number.isFinite(point.dominance) && Number.isFinite(asset.dominance)
    ? (dominanceWeight * (point.dominance - asset.dominance)) ** 2
    : 0;
  return Math.sqrt(planar + dominance);
}

function recencyPenalty(id, recent, penalty) {
  const position = recent.indexOf(id);
  // The most recent winner gets the full penalty, older ones progressively less.
  return position < 0 ? 0 : penalty * (1 - position / recent.length);
}

export function rankReactionCandidates(assets, valence, arousal, limit = 5, options = {}) {
  if (!Array.isArray(assets)) return [];
  const {
    dominance = null,
    dominanceWeight = DEFAULT_DOMINANCE_WEIGHT,
    recent = [],
    recencyPenalty: penalty = DEFAULT_RECENCY_PENALTY,
  } = options;
  const point = { valence, arousal, dominance };
  const candidates = assets
    .filter(asset => Number.isFinite(asset.valence) && Number.isFinite(asset.arousal) && asset.publicPath)
    .map(asset => {
      const distance = reactionDistance(asset, point, dominanceWeight);
      return { asset, distance, score: distance + recencyPenalty(asset.id, recent, penalty) };
    });

  candidates.sort((left, right) => {
    const scoreOrder = left.score - right.score;
    if (Math.abs(scoreOrder) > 1e-12) return scoreOrder;
    if (left.asset.id < right.asset.id) return -1;
    if (left.asset.id > right.asset.id) return 1;
    return 0;
  });

  return candidates.slice(0, limit).map(candidate => ({
    asset: { ...candidate.asset },
    distance: Number(candidate.distance.toFixed(6)),
  }));
}

export function selectNearestReaction(assets, valence, arousal, options = {}) {
  if (!Array.isArray(assets) || assets.length === 0) {
    return { asset: null, fallbackReason: 'No usable reaction assets are available.' };
  }

  const candidates = rankReactionCandidates(assets, valence, arousal, 5, options);

  if (candidates.length === 0) {
    return { asset: null, fallbackReason: 'The reaction manifest has no positioned browser assets.' };
  }

  const winner = candidates[0];
  return {
    asset: { ...winner.asset },
    distance: winner.distance,
    candidates,
    fallbackReason: null,
  };
}

// Server-ranked matches compare the reply and each reaction's descriptions in activation space,
// which retrieves the intended reaction far more often than coordinates do. Coordinates remain
// the fallback when no reaction embeddings have been built.
export function selectMatchedReaction(assets, matches, options = {}) {
  const { recent = [], recencyPenalty: penalty = DEFAULT_RECENCY_PENALTY } = options;
  if (!Array.isArray(assets) || !Array.isArray(matches) || matches.length === 0) return null;
  const usable = new Map(assets.filter(asset => asset.publicPath).map(asset => [asset.id, asset]));
  const candidates = matches
    .filter(match => usable.has(match.id) && Number.isFinite(match.similarity))
    .map(match => ({
      asset: { ...usable.get(match.id) },
      similarity: match.similarity,
      // Reported as a distance so diagnostics read the same for both selection methods.
      distance: Number((1 - match.similarity).toFixed(6)),
      score: -match.similarity + recencyPenalty(match.id, recent, penalty),
    }));
  if (candidates.length === 0) return null;
  candidates.sort((left, right) => left.score - right.score || (left.asset.id < right.asset.id ? -1 : 1));
  const ranked = candidates.slice(0, 5).map(({ score, ...candidate }) => candidate);
  return {
    asset: { ...ranked[0].asset },
    distance: ranked[0].distance,
    candidates: ranked,
    method: 'activation',
    fallbackReason: null,
  };
}

// How a turn's readings are shaped, for picking a meme format later:
// one clear feeling, two competing feelings, or a reply that drifted from its plan.
export const SPLIT_WEIGHT = 0.3;
export const DRIFT_DISTANCE = 0.6;

export function readoutShape(result) {
  const modes = result?.affect?.modes || [];
  // Both readings use the example-built prototypes, so a gap reflects the reply, not the method.
  const planned = result?.spans?.boundary;
  const written = result?.spans?.reply?.mean;
  if (planned && written) {
    const drift = Math.hypot(planned.valence - written.valence, planned.arousal - written.arousal);
    if (drift >= DRIFT_DISTANCE) return { kind: 'drift', from: planned, to: written, distance: Number(drift.toFixed(4)) };
  }
  if (modes.length >= 2 && modes[1].weight >= SPLIT_WEIGHT) return { kind: 'split', modes: modes.slice(0, 2) };
  return { kind: 'single' };
}
