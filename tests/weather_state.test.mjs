import assert from 'node:assert/strict';
import test from 'node:test';

import { WeatherHistory } from '../frontend/weather-state.mjs';

const snapshot = (hex, valence) => ({
  affect: { valence, arousal: 0.2, confidence: 0.7, matches: [{ concept: 'calm', similarity: 0.5 }] },
  color: { hex, raw_rgb: [1, 2, 3] },
  layer: 24,
  reaction: {
    asset: { id: `reaction-${hex}`, publicPath: `/${hex}.gif` },
    distance: 0.1,
    candidates: [{ asset: { id: `reaction-${hex}`, publicPath: `/${hex}.gif` }, distance: 0.1 }],
  },
  generation: { debug_id: `debug-${hex}`, stop_reason: 'eos', generated_token_count: 12, server_frontend_text_match: true },
});

test('selecting history displays its saved snapshot without changing live state', () => {
  const displayed = [];
  const history = new WeatherHistory(value => displayed.push(value));
  const first = snapshot('#111111', -0.5);
  history.add('one', first);
  history.add('two', snapshot('#222222', 0.5));

  first.color.hex = '#ffffff';
  assert.equal(history.select('one'), true);
  assert.equal(displayed.at(-1).color.hex, '#111111');
  assert.equal(displayed.at(-1).reaction.asset.id, 'reaction-#111111');
  assert.equal(displayed.at(-1).generation.debug_id, 'debug-#111111');
  assert.equal(displayed.at(-1).reaction.candidates[0].asset.id, 'reaction-#111111');
  assert.deepEqual(history.state(), { latestId: 'two', selectedId: 'one', isLive: false });
});

test('return to live restores newest weather and a new response becomes live', () => {
  let displayed;
  const history = new WeatherHistory(value => { displayed = value; });
  history.add('one', snapshot('#111111', -0.5));
  history.add('two', snapshot('#222222', 0.5));
  history.select('one');

  assert.equal(history.returnToLive(), true);
  assert.equal(displayed.color.hex, '#222222');
  assert.equal(displayed.reaction.asset.id, 'reaction-#222222');
  assert.equal(history.state().isLive, true);

  history.select('one');
  history.add('three', snapshot('#333333', 0.8));
  assert.equal(displayed.color.hex, '#333333');
  assert.equal(displayed.reaction.asset.id, 'reaction-#333333');
  assert.equal(displayed.generation.debug_id, 'debug-#333333');
  assert.deepEqual(history.state(), { latestId: 'three', selectedId: 'three', isLive: true });
});

test('unknown history selection leaves the displayed snapshot unchanged', () => {
  let displayed;
  const history = new WeatherHistory(value => { displayed = value; });
  history.add('one', snapshot('#111111', 0));
  assert.equal(history.select('missing'), false);
  assert.equal(displayed.color.hex, '#111111');
});
