function freezeCopy(value) {
  const copy = JSON.parse(JSON.stringify(value));
  const deepFreeze = item => {
    if (item && typeof item === 'object' && !Object.isFrozen(item)) {
      Object.values(item).forEach(deepFreeze);
      Object.freeze(item);
    }
    return item;
  };
  return deepFreeze(copy);
}

/** Keeps historical display selection separate from the newest live snapshot. */
export class WeatherHistory {
  constructor(onDisplay, onSelectionChange = () => {}) {
    this.snapshots = new Map();
    this.latestId = null;
    this.selectedId = null;
    this.onDisplay = onDisplay;
    this.onSelectionChange = onSelectionChange;
  }

  add(id, snapshot) {
    const saved = freezeCopy(snapshot);
    this.snapshots.set(id, saved);
    this.latestId = id;
    this.selectedId = id;
    this.onDisplay(saved);
    this.onSelectionChange(this.state());
    return saved;
  }

  select(id) {
    const snapshot = this.snapshots.get(id);
    if (!snapshot) return false;
    this.selectedId = id;
    this.onDisplay(snapshot);
    this.onSelectionChange(this.state());
    return true;
  }

  returnToLive() {
    if (this.latestId === null) return false;
    return this.select(this.latestId);
  }

  state() {
    return {
      latestId: this.latestId,
      selectedId: this.selectedId,
      isLive: this.latestId !== null && this.selectedId === this.latestId,
    };
  }
}
