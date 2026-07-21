export function rankReactionCandidates(assets, valence, arousal, limit = 5) {
  if (!Array.isArray(assets)) return [];
  const candidates = assets
    .filter(asset => Number.isFinite(asset.valence) && Number.isFinite(asset.arousal) && asset.publicPath)
    .map(asset => ({
      asset,
      distance: Math.hypot(valence - asset.valence, arousal - asset.arousal),
    }));

  candidates.sort((left, right) => {
    const distanceOrder = left.distance - right.distance;
    if (Math.abs(distanceOrder) > 1e-12) return distanceOrder;
    if (left.asset.id < right.asset.id) return -1;
    if (left.asset.id > right.asset.id) return 1;
    return 0;
  });

  return candidates.slice(0, limit).map(candidate => ({
    asset: { ...candidate.asset },
    distance: Number(candidate.distance.toFixed(6)),
  }));
}

export function selectNearestReaction(assets, valence, arousal) {
  if (!Array.isArray(assets) || assets.length === 0) {
    return { asset: null, fallbackReason: 'No usable reaction assets are available.' };
  }

  const candidates = rankReactionCandidates(assets, valence, arousal, 5);

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
