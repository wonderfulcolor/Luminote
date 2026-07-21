import assert from 'node:assert/strict';
import test from 'node:test';

import { buildManifestHealth } from '../frontend/reaction-diagnostics.mjs';

test('manifest health counts active, skipped HEIC, unmatched, duplicate, and alpha groups', () => {
  const health = buildManifestHealth({
    version: 2,
    assets: [
      { id: 'transparent', format: 'png', valence: 0, arousal: 0, hasTransparentPixels: true },
      { id: 'opaque', format: 'webp', valence: 0, arousal: 0, hasTransparentPixels: false },
      { id: 'jpeg', format: 'jpg', valence: 1, arousal: 1, hasTransparentPixels: false },
    ],
    skippedAssets: [{ filename: 'one.HEIC' }, { filename: 'two.HEICS' }],
    unmatchedAssets: [{ filename: 'missing.png', reason: 'unmatched' }],
  });
  assert.equal(health.activeCount, 3);
  assert.equal(health.skippedHeicCount, 2);
  assert.equal(health.unmatchedCount, 1);
  assert.deepEqual(health.exactCoordinateDuplicateGroups[0].assetIds, ['opaque', 'transparent']);
  assert.deepEqual(health.transparentActiveAssets, ['transparent']);
  assert.deepEqual(health.opaquePngWebpActiveAssets, ['opaque']);
});
