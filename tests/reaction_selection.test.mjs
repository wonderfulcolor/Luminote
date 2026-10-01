import assert from 'node:assert/strict';
import test from 'node:test';

import { rankReactionCandidates, readoutShape, selectMatchedReaction, selectNearestReaction } from '../frontend/reaction-selection.mjs';

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

test('dominance separates reactions that share valence and arousal', () => {
  const assets = [
    { ...asset('plotting', -0.7, 0.8), dominance: 0.6 },
    { ...asset('cornered', -0.7, 0.8), dominance: -0.6 },
  ];
  assert.equal(selectNearestReaction(assets, -0.7, 0.8, { dominance: 0.5 }).asset.id, 'plotting');
  assert.equal(selectNearestReaction(assets, -0.7, 0.8, { dominance: -0.5 }).asset.id, 'cornered');
  // Without a dominance reading the original alphabetical tie-break still applies.
  assert.equal(selectNearestReaction(assets, -0.7, 0.8).asset.id, 'cornered');
});

test('assets without dominance are still ranked on valence and arousal', () => {
  const assets = [asset('legacy', 0.4, 0.4), { ...asset('placed', 0.4, 0.4), dominance: 0.9 }];
  const ranked = rankReactionCandidates(assets, 0.4, 0.4, 5, { dominance: -0.9 });
  assert.deepEqual(ranked.map(candidate => candidate.asset.id), ['legacy', 'placed']);
  assert.equal(ranked[0].distance, 0);
});

test('recently shown reactions yield to close alternatives but not distant ones', () => {
  const assets = [asset('again', 0, 0), asset('close', 0.1, 0), asset('distant', 0.9, 0)];
  assert.equal(selectNearestReaction(assets, 0, 0, { recent: ['again'] }).asset.id, 'close');
  assert.equal(selectNearestReaction([assets[0], assets[2]], 0, 0, { recent: ['again'] }).asset.id, 'again');
  // The reported distance stays the true distance; only the ranking is penalised.
  assert.equal(selectNearestReaction(assets, 0, 0, { recent: ['again'] }).candidates[1].distance, 0);
});

test('readout shape distinguishes single, split, and drifted turns', () => {
  const mode = (weight, valence) => ({ concepts: ['x'], valence, arousal: 0, dominance: null, weight });
  assert.equal(readoutShape({ affect: { modes: [mode(0.9, 0.5)] } }).kind, 'single');
  assert.equal(readoutShape({ affect: { modes: [mode(0.55, 0.5), mode(0.45, -0.5)] } }).kind, 'split');
  const drifted = readoutShape({
    affect: { modes: [mode(1, 0.5)] },
    spans: { boundary: { valence: 0.6, arousal: 0.2 }, reply: { mean: { valence: -0.4, arousal: 0.3 } } },
  });
  assert.equal(drifted.kind, 'drift');
  assert.ok(drifted.distance > 0.9);
});

test('activation matches pick the most similar usable reaction and respect recency', () => {
  const assets = [asset('fine', 0, 0), asset('panic', 0, 0), { ...asset('unpublished', 0, 0), publicPath: null }];
  const matches = [
    { id: 'unpublished', similarity: 0.9 }, { id: 'fine', similarity: 0.6 }, { id: 'panic', similarity: 0.55 },
  ];
  const selection = selectMatchedReaction(assets, matches);
  assert.equal(selection.asset.id, 'fine');
  assert.equal(selection.method, 'activation');
  assert.equal(selection.candidates[0].distance, 0.4);
  assert.equal(selectMatchedReaction(assets, matches, { recent: ['fine'] }).asset.id, 'panic');
});

test('without usable matches, activation selection defers to coordinates', () => {
  assert.equal(selectMatchedReaction([asset('a', 0, 0)], []), null);
  assert.equal(selectMatchedReaction([asset('a', 0, 0)], undefined), null);
  assert.equal(selectMatchedReaction([asset('a', 0, 0)], [{ id: 'missing', similarity: 1 }]), null);
});
