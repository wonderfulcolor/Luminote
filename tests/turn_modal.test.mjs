import assert from 'node:assert/strict';
import test from 'node:test';

import { createTurnModalController } from '../frontend/turn-modal.mjs';

class FakeTarget {
  constructor() { this.listeners = new Map(); this.open = false; }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  emit(type, event = { target: this }) { this.listeners.get(type)?.(event); }
  showModal() { this.open = true; }
  close() { this.open = false; }
}

test('shared turn modal renders the exact immutable snapshot and close button works', () => {
  const dialog = new FakeTarget();
  const closeButton = new FakeTarget();
  let rendered;
  const controller = createTurnModalController({ dialog, closeButton, render: snapshot => { rendered = snapshot; } });
  const snapshot = Object.freeze({ affect: Object.freeze({ valence: -0.4, arousal: 0.2 }), reaction: Object.freeze({ asset: Object.freeze({ id: 'processing' }) }) });
  controller.open(snapshot);
  assert.equal(rendered, snapshot);
  assert.equal(controller.currentSnapshot(), snapshot);
  assert.equal(dialog.open, true);
  closeButton.emit('click');
  assert.equal(dialog.open, false);
  assert.equal(controller.currentSnapshot(), null);
});

test('backdrop and Escape cancellation clear shared modal state', () => {
  const dialog = new FakeTarget();
  const closeButton = new FakeTarget();
  const controller = createTurnModalController({ dialog, closeButton, render: () => {} });
  controller.open({ id: 'backdrop' });
  dialog.emit('click', { target: dialog });
  assert.equal(dialog.open, false);
  controller.open({ id: 'escape' });
  dialog.emit('keydown', { key: 'Escape', preventDefault() {} });
  assert.equal(dialog.open, false);
  assert.equal(controller.currentSnapshot(), null);
});
