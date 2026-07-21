import assert from 'node:assert/strict';
import test from 'node:test';

import { rankReactionCandidates, selectNearestReaction } from '../frontend/reaction-selection.mjs';

const asset = (id, valence, arousal) => ({
  id, valence, arousal, publicPath: `/reaction-assets/${id}.png`, filename: `${id}.png`,
});

test('nearest reaction uses Euclidean distance', () => {
  const selection = selectNearestReaction([
    asset('far', -0.8, -0.8), asset('near', 0.4, 0.5),
  ], 0.35, 0.45);
  assert.equal(selection.asset.id, 'near');
  assert.ok(selection.distance < 0.08);
  assert.deepEqual(selection.candidates.map(candidate => candidate.asset.id), ['near', 'far']);
});

test('equal distances use asset ID alphabetical order', () => {
  const selection = selectNearestReaction([
    asset('zeta', 1, 0), asset('alpha', -1, 0),
  ], 0, 0);
  assert.equal(selection.asset.id, 'alpha');
});

test('empty or unusable catalogs return a graceful fallback', () => {
  assert.equal(selectNearestReaction([], 0, 0).asset, null);
  assert.match(selectNearestReaction([], 0, 0).fallbackReason, /No usable/);
  assert.equal(selectNearestReaction([{ id: 'broken' }], 0, 0).asset, null);
});

test('top five ranking uses distance then the existing alphabetical tie-break', () => {
  const assets = [
    asset('zeta', 1, 0), asset('alpha', -1, 0), asset('near', 0.1, 0),
    asset('fourth', 0.4, 0), asset('fifth', 0.5, 0), asset('sixth', 0.6, 0),
  ];
  const ranked = rankReactionCandidates(assets, 0, 0);
  assert.deepEqual(ranked.map(candidate => candidate.asset.id), ['near', 'fourth', 'fifth', 'sixth', 'alpha']);
  assert.equal(ranked[4].distance, 1);
});

test('inactive or malformed entries cannot enter ranking and diagnostics do not change the winner', () => {
  const assets = [asset('winner', 0.1, 0.1), asset('runner-up', 0.2, 0.2), { ...asset('inactive', 0, 0), publicPath: null }];
  const before = selectNearestReaction(assets, 0, 0);
  const diagnostics = rankReactionCandidates(assets, 0, 0);
  const after = selectNearestReaction(assets, 0, 0);
  assert.equal(before.asset.id, 'winner');
  assert.equal(after.asset.id, 'winner');
  assert.deepEqual(diagnostics.map(candidate => candidate.asset.id), ['winner', 'runner-up']);
});
