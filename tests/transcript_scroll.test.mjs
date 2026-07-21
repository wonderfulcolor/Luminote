import assert from 'node:assert/strict';
import test from 'node:test';

import { isNearBottom } from '../frontend/transcript-scroll.mjs';

test('new output follows a transcript already near its bottom', () => {
  assert.equal(isNearBottom({ scrollHeight: 1000, scrollTop: 620, clientHeight: 300 }), true);
});

test('new output does not yank a user inspecting older responses', () => {
  assert.equal(isNearBottom({ scrollHeight: 1000, scrollTop: 200, clientHeight: 300 }), false);
});
