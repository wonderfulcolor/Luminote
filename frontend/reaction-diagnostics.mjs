export function buildManifestHealth(manifest) {
  const assets = Array.isArray(manifest?.assets) ? manifest.assets : [];
  const skipped = Array.isArray(manifest?.skippedAssets) ? manifest.skippedAssets : [];
  const unmatched = Array.isArray(manifest?.unmatchedAssets) ? manifest.unmatchedAssets : [];
  const preflight = manifest?.preflight || {};
  const groups = new Map();
  assets.forEach(asset => {
    if (!Number.isFinite(asset.valence) || !Number.isFinite(asset.arousal)) return;
    const key = `${asset.valence}|${asset.arousal}`;
    const group = groups.get(key) || { valence: asset.valence, arousal: asset.arousal, assetIds: [] };
    group.assetIds.push(asset.id);
    groups.set(key, group);
  });
  const derivedDuplicates = [...groups.values()]
    .filter(group => group.assetIds.length > 1)
    .map(group => ({ ...group, assetIds: [...group.assetIds].sort() }));
  return {
    version: manifest?.version ?? null,
    activeCount: assets.length,
    skippedHeicCount: skipped.filter(item => /\.heics?$/i.test(item.filename || '')).length,
    unmatchedCount: unmatched.length,
    unmatchedAssets: unmatched.map(item => ({ ...item })),
    exactCoordinateDuplicateGroups: Array.isArray(preflight.exactCoordinateDuplicateGroups)
      ? preflight.exactCoordinateDuplicateGroups.map(group => ({ ...group, assetIds: [...group.assetIds] }))
      : derivedDuplicates,
    transparentActiveAssets: assets.filter(asset => asset.hasTransparentPixels === true).map(asset => asset.id),
    opaquePngWebpActiveAssets: assets
      .filter(asset => ['png', 'webp'].includes(String(asset.format).toLowerCase()) && asset.hasTransparentPixels === false)
      .map(asset => asset.id),
  };
}
