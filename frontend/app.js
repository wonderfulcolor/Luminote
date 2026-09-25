import { buildManifestHealth } from './reaction-diagnostics.mjs';
import { readoutShape, selectMatchedReaction, selectNearestReaction } from './reaction-selection.mjs';
import { isNearBottom } from './transcript-scroll.mjs';
import { createTurnModalController } from './turn-modal.mjs';
import { WeatherHistory } from './weather-state.mjs';

const transcript = document.querySelector('#transcript');
const form = document.querySelector('#composer');
const prompt = document.querySelector('#prompt');
const button = form.querySelector('button');
const liveWeatherButton = document.querySelector('#live-weather');
const diagnosticsDialog = document.querySelector('#turn-diagnostics-dialog');
const modalContent = document.querySelector('#turn-modal-content');
const layerInput = document.querySelector('#layer');
const messages = [];
// Most recent winner first; lets selection avoid showing the same reaction every turn.
const recentReactionIds = [];
const RECENT_REACTION_LIMIT = 4;
let weatherSequence = 0;

const reactionManifestPromise = fetch('/reaction-metadata/reaction_manifest.json')
  .then(response => {
    if (!response.ok) throw new Error(`Reaction manifest unavailable (${response.status})`);
    return response.json();
  })
  .catch(error => ({ assets: [], skippedAssets: [], unmatchedAssets: [], loadError: error.message }));

function formatTextMatch(value) {
  if (value === true) return 'Yes';
  if (value === false) return 'NO — inspect local log';
  return 'Unavailable';
}

function formatCoordinate(valence, arousal) {
  return Number.isFinite(valence) && Number.isFinite(arousal)
    ? `(${valence.toFixed(3)}, ${arousal.toFixed(3)})`
    : '—';
}

function addDetail(list, label, value, alert = false) {
  const row = document.createElement('div');
  if (alert) row.className = 'integrity-alert';
  const term = document.createElement('dt');
  const description = document.createElement('dd');
  term.textContent = label;
  description.textContent = value;
  row.append(term, description);
  list.append(row);
}

function createReactionFigure(reaction, className, compact = false) {
  const asset = reaction?.asset;
  if (!asset) return null;
  const figure = document.createElement('figure');
  figure.className = className;
  const image = document.createElement('img');
  image.src = asset.publicPath;
  image.alt = `Mapped reaction: ${asset.label || asset.id}`;
  const caption = document.createElement('figcaption');
  caption.textContent = compact ? (asset.label || asset.id) : `Nearest mapped reaction: ${asset.label || asset.id}`;
  figure.append(image, caption);
  return figure;
}

function createCandidateTable(snapshot, extraClass = '') {
  const section = document.createElement('section');
  section.className = `candidate-diagnostics ${extraClass}`.trim();
  const heading = document.createElement('h3');
  heading.textContent = 'Five nearest active reactions';
  section.append(heading);
  const candidates = snapshot.reaction?.candidates || [];
  if (candidates.length === 0) {
    const empty = document.createElement('p');
    empty.textContent = 'No positioned active candidates were available.';
    section.append(empty);
    return section;
  }

  const selectedId = snapshot.reaction?.asset?.id;
  if (candidates[0]?.asset?.id !== selectedId) {
    const note = document.createElement('p');
    note.className = 'preserved-selection-note';
    note.textContent = `Preserved historical winner: ${selectedId || 'unavailable'}. Rank one reflects the candidates saved with this snapshot.`;
    section.append(note);
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'candidate-table-wrap';
  const table = document.createElement('table');
  table.className = 'candidate-table';
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  ['Rank', 'Asset', 'Original file', 'Coordinate', 'Distance', 'Source', 'Type', 'Selected'].forEach(label => {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = label;
    headRow.append(cell);
  });
  head.append(headRow);
  const body = document.createElement('tbody');
  candidates.forEach((candidate, index) => {
    const row = document.createElement('tr');
    const selected = candidate.asset.id === selectedId;
    if (selected) row.className = 'candidate-winner';
    const values = [
      String(index + 1),
      candidate.asset.id,
      candidate.asset.filename,
      formatCoordinate(candidate.asset.valence, candidate.asset.arousal),
      candidate.distance.toFixed(4),
      String(candidate.asset.coordinateSource || '—').toUpperCase(),
      candidate.asset.animated ? 'Animated' : 'Static',
      selected ? 'Winner' : '—',
    ];
    values.forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  });
  table.append(head, body);
  wrapper.append(table);
  section.append(wrapper);
  return section;
}

function createManifestHealth(snapshot, expanded = false) {
  const health = snapshot.manifestHealth || {};
  const section = document.createElement('section');
  section.className = `manifest-health ${expanded ? 'manifest-health-expanded' : ''}`.trim();
  const heading = document.createElement('h3');
  heading.textContent = 'Reaction manifest health';
  const list = document.createElement('dl');
  addDetail(list, 'Manifest version', health.version == null ? '—' : String(health.version));
  addDetail(list, 'Active assets', String(health.activeCount ?? 0));
  addDetail(list, 'Skipped HEIC/HEICS', String(health.skippedHeicCount ?? 0));
  addDetail(list, 'Supported unmatched', String(health.unmatchedCount ?? 0));
  addDetail(list, 'Duplicate-coordinate groups', String(health.exactCoordinateDuplicateGroups?.length ?? 0));
  section.append(heading, list);

  const duplicates = health.exactCoordinateDuplicateGroups || [];
  const duplicateText = document.createElement('p');
  duplicateText.textContent = duplicates.length
    ? `Exact duplicates: ${duplicates.map(group => `${formatCoordinate(group.valence, group.arousal)} — ${group.assetIds.join(', ')}`).join('; ')}`
    : 'Exact duplicates: none.';
  section.append(duplicateText);
  const unmatched = health.unmatchedAssets || [];
  const unmatchedText = document.createElement('p');
  unmatchedText.textContent = unmatched.length
    ? `Supported but unmatched: ${unmatched.map(asset => asset.filename).join(', ')}.`
    : 'Supported but unmatched: none.';
  section.append(unmatchedText);

  if (expanded) {
    const transparent = document.createElement('p');
    transparent.textContent = `Real transparent pixels: ${(health.transparentActiveAssets || []).join(', ') || 'none detected'}.`;
    const opaque = document.createElement('p');
    opaque.textContent = `PNG/WebP with no transparent pixels (manual review): ${(health.opaquePngWebpActiveAssets || []).join(', ') || 'none'}.`;
    const warning = document.createElement('p');
    warning.className = 'transparency-note';
    warning.textContent = 'A checkerboard baked into RGB pixels is part of the source image and cannot be removed by CSS.';
    section.append(transparent, opaque, warning);
  }
  return section;
}

function createPrototypeMatches(snapshot) {
  const section = document.createElement('section');
  section.className = 'turn-prototype-matches';
  const heading = document.createElement('h3');
  heading.textContent = 'Top prototype matches';
  const list = document.createElement('ol');
  (snapshot.affect?.matches || []).forEach(match => {
    const item = document.createElement('li');
    item.textContent = `${match.concept} · ${match.similarity.toFixed(3)}`;
    list.append(item);
  });
  section.append(heading, list);
  return section;
}

function describeReading(estimate) {
  if (!estimate) return '—';
  const modes = (estimate.modes || []).map(mode => mode.concepts.join('/')).join(' vs ');
  const label = modes || estimate.matches?.[0]?.concept || '—';
  return `${label} ${formatCoordinate(estimate.valence, estimate.arousal)}`;
}

function describeTarget(target) {
  if (!target) return '—';
  return Object.entries(target)
    .sort((left, right) => right[1] - left[1])
    .map(([name, score]) => `${name} ${Math.round(score * 100)}%`)
    .join(' · ');
}

function describeSelection(snapshot) {
  if (snapshot.reaction?.method === 'activation') {
    return `Activation match · similarity ${snapshot.reaction.candidates[0]?.similarity?.toFixed(3) ?? '—'}`;
  }
  return `Coordinates${snapshot.reaction_match?.reason ? ` · ${snapshot.reaction_match.reason}` : ''}`;
}

function describeShape(shape) {
  if (shape.kind === 'drift') return `Drift: planned ${describeReading(shape.from)} → wrote ${describeReading(shape.to)}`;
  if (shape.kind === 'split') return `Split: ${shape.modes.map(mode => mode.concepts.join('/')).join(' vs ')}`;
  return 'Single feeling';
}

function createSpanReadings(snapshot) {
  const section = document.createElement('section');
  section.className = 'turn-prototype-matches turn-span-readings';
  const heading = document.createElement('h3');
  heading.textContent = `Per-token readings · layer ${snapshot.spans?.layer ?? '—'}`;
  const list = document.createElement('dl');
  list.className = 'response-detail-values';
  const spans = snapshot.spans;
  if (!spans || spans.error) {
    addDetail(list, 'Unavailable', spans?.error || 'This turn has no span readings.');
    section.append(heading, list);
    return section;
  }
  const reply = spans.reply;
  const peaks = (reply?.peaks || [])
    .map(peak => `${peak.concept} at ${JSON.stringify(spans.reply_tokens?.[peak.token_index] ?? '?')}`)
    .join(' · ');
  addDetail(list, 'Reading the user', describeReading(spans.user?.mean));
  addDetail(list, 'Planned (boundary)', describeReading(spans.boundary));
  addDetail(list, 'While replying', describeReading(reply?.mean));
  addDetail(list, 'Reply peaks', peaks || '—');
  addDetail(list, 'Reply is about', describeTarget(reply?.mean?.target));
  addDetail(list, 'Shape', describeShape(snapshot.shape || readoutShape(snapshot)));
  if (!spans.example_prototypes) addDetail(list, 'Note', 'Calibrated without examples; re-run calibration.', true);
  section.append(heading, list);
  return section;
}

function positionMarker(marker, valence, arousal) {
  marker.style.left = `${(valence + 1) * 50}%`;
  marker.style.top = `${(1 - arousal) * 50}%`;
}

function createLargeMap(snapshot) {
  const wrap = document.createElement('div');
  wrap.className = 'large-map-wrap';
  const image = document.createElement('img');
  image.src = '/assets/public_affective_map.png';
  image.alt = 'Affective circumplex, enlarged';
  const measured = document.createElement('i');
  measured.className = 'map-marker measured-marker';
  measured.textContent = 'M';
  measured.title = `Measured point ${formatCoordinate(snapshot.affect.valence, snapshot.affect.arousal)}`;
  positionMarker(measured, snapshot.affect.valence, snapshot.affect.arousal);
  wrap.append(image, measured);
  (snapshot.reaction?.candidates || []).forEach((candidate, index) => {
    const marker = document.createElement('i');
    marker.className = `map-marker candidate-marker ${candidate.asset.id === snapshot.reaction?.asset?.id ? 'winner-marker' : ''}`.trim();
    marker.textContent = String(index + 1);
    marker.title = `${index + 1}. ${candidate.asset.id} ${formatCoordinate(candidate.asset.valence, candidate.asset.arousal)}`;
    positionMarker(marker, candidate.asset.valence, candidate.asset.arousal);
    wrap.append(marker);
  });
  const legend = document.createElement('p');
  legend.className = 'map-legend';
  legend.textContent = 'M = measured point · 1–5 = nearest active reaction coordinates · outlined marker = saved winner';
  const container = document.createElement('div');
  container.append(wrap, legend);
  return container;
}

function renderTurnModal(snapshot) {
  const grid = document.createElement('div');
  grid.className = 'expanded-grid';
  grid.append(createLargeMap(snapshot));
  const details = document.createElement('div');
  details.className = 'expanded-details';
  const reaction = createReactionFigure(snapshot.reaction, 'expanded-reaction', true);
  if (reaction) details.append(reaction);
  const list = document.createElement('dl');
  const asset = snapshot.reaction?.asset;
  addDetail(list, 'Valence', snapshot.affect.valence.toFixed(3));
  addDetail(list, 'Arousal', snapshot.affect.arousal.toFixed(3));
  addDetail(list, 'Confidence', `${Math.round(snapshot.affect.confidence * 100)}%`);
  addDetail(list, 'Selected reaction asset', asset?.id || '—');
  addDetail(list, 'Selected asset coordinate', asset ? formatCoordinate(asset.valence, asset.arousal) : '—');
  addDetail(list, 'Distance to selected asset', Number.isFinite(snapshot.reaction?.distance) ? snapshot.reaction.distance.toFixed(4) : '—');
  addDetail(list, 'Coordinate source', String(asset?.coordinateSource || '—').toUpperCase());
  addDetail(list, 'Selected by', describeSelection(snapshot));
  addDetail(list, 'Generation debug ID', snapshot.generation?.debug_id || '—');
  details.append(list);
  grid.append(details);
  modalContent.replaceChildren(grid, createCandidateTable(snapshot, 'modal-candidates'), createManifestHealth(snapshot, true));
}

const turnModal = createTurnModalController({
  dialog: diagnosticsDialog,
  closeButton: document.querySelector('#close-turn-diagnostics'),
  render: renderTurnModal,
});

function createResponseDetails(snapshot) {
  const details = document.createElement('details');
  details.className = 'response-details';
  details.addEventListener('click', event => event.stopPropagation());
  details.addEventListener('keydown', event => event.stopPropagation());
  const summary = document.createElement('summary');
  summary.textContent = 'Turn details';
  const body = document.createElement('div');
  body.className = 'response-details-body';
  const overview = document.createElement('div');
  overview.className = 'response-details-grid';

  const miniReaction = createReactionFigure(snapshot.reaction, 'response-detail-reaction', true);
  if (miniReaction) overview.append(miniReaction);

  const map = document.createElement('button');
  map.type = 'button';
  map.className = 'response-mini-map';
  map.setAttribute('aria-label', 'Open enlarged affective circumplex for this response.');
  const mapImage = document.createElement('img');
  mapImage.src = '/assets/public_affective_map.png';
  mapImage.alt = 'Miniature affective circumplex';
  const point = document.createElement('i');
  point.className = 'map-point response-map-point';
  positionMarker(point, snapshot.affect.valence, snapshot.affect.arousal);
  const expandLabel = document.createElement('span');
  expandLabel.textContent = 'Enlarge';
  map.append(mapImage, point, expandLabel);
  map.addEventListener('click', event => {
    event.stopPropagation();
    turnModal.open(snapshot);
  });
  map.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      event.stopPropagation();
      turnModal.open(snapshot);
    }
  });
  overview.append(map);

  const list = document.createElement('dl');
  list.className = 'response-detail-values';
  const asset = snapshot.reaction?.asset;
  addDetail(list, 'Valence', snapshot.affect.valence.toFixed(3));
  addDetail(list, 'Arousal', snapshot.affect.arousal.toFixed(3));
  addDetail(list, 'Dominance', Number.isFinite(snapshot.affect.dominance) ? snapshot.affect.dominance.toFixed(3) : '—');
  addDetail(list, 'Confidence', `${Math.round(snapshot.affect.confidence * 100)}%`);
  addDetail(list, 'Weather from', snapshot.weather_source === 'reply' ? 'Reply as written' : 'Response boundary');
  addDetail(list, 'Color', `${snapshot.color.hex} · rgb(${snapshot.color.raw_rgb.join(', ')})`);
  addDetail(list, 'Layer', String(snapshot.layer));
  addDetail(list, 'Boundary token', `${JSON.stringify(snapshot.response_start_token)} (${snapshot.response_start_token_id})`);
  addDetail(list, 'Model', snapshot.model || '—');
  addDetail(list, 'Asset ID', asset?.id || '—');
  addDetail(list, 'Original filename', asset?.filename || '—');
  addDetail(list, 'File format', String(asset?.format || '—').toUpperCase());
  addDetail(list, 'Real transparent pixels', asset?.hasTransparentPixels === true ? 'Yes' : 'No');
  addDetail(list, 'Asset coordinate', asset ? formatCoordinate(asset.valence, asset.arousal) : '—');
  addDetail(list, 'Distance', Number.isFinite(snapshot.reaction?.distance) ? snapshot.reaction.distance.toFixed(4) : '—');
  addDetail(list, 'Coordinate source', String(asset?.coordinateSource || '—').toUpperCase());
  addDetail(list, 'Selected by', describeSelection(snapshot));
  addDetail(list, 'Asset type', asset ? (asset.animated ? 'Animated' : 'Static') : '—');
  addDetail(list, 'Generation debug ID', snapshot.generation?.debug_id || '—');
  addDetail(list, 'Stop reason', snapshot.generation?.stop_reason || '—');
  addDetail(list, 'Generated tokens', Number.isInteger(snapshot.generation?.generated_token_count) ? String(snapshot.generation.generated_token_count) : '—');
  addDetail(list, 'Server/frontend match', formatTextMatch(snapshot.generation?.server_frontend_text_match), snapshot.generation?.server_frontend_text_match === false);
  overview.append(list);
  body.append(overview, createSpanReadings(snapshot), createPrototypeMatches(snapshot), createCandidateTable(snapshot), createManifestHealth(snapshot));
  details.append(summary, body);
  return details;
}

function updateWeather(result) {
  document.querySelector('#weather-layer').style.backgroundColor = result.color.hex;
  document.querySelector('#weather-dot').style.backgroundColor = result.color.hex;
  layerInput.max = result.layer_count - 1;
  layerInput.value = result.layer;
}

function updateSelection({ selectedId, isLive, latestId }) {
  document.querySelectorAll('.message.assistant[data-weather-id]').forEach(message => {
    const selected = message.dataset.weatherId === selectedId;
    message.classList.toggle('selected-weather', selected);
    message.setAttribute('aria-pressed', String(selected));
  });
  liveWeatherButton.disabled = latestId === null || isLive;
  liveWeatherButton.textContent = isLive ? 'Latest weather' : 'Return to live weather';
}

const weatherHistory = new WeatherHistory(updateWeather, updateSelection);

function isNearTranscriptBottom() {
  return isNearBottom(transcript);
}

function scrollTranscriptToBottom(behavior = 'smooth') {
  transcript.scrollTo({ top: transcript.scrollHeight, behavior });
}

function addMessage(role, text, extra = '', keepAtBottom = isNearTranscriptBottom()) {
  const article = document.createElement('article');
  article.className = `message ${role} ${extra}`;
  const paragraph = document.createElement('p');
  paragraph.textContent = text;
  article.append(paragraph);
  transcript.append(article);
  if (keepAtBottom) requestAnimationFrame(() => scrollTranscriptToBottom());
  return article;
}

function attachWeatherSnapshot(article, snapshot, keepAtBottom) {
  const paragraph = article.querySelector('p');
  const content = document.createElement('div');
  content.className = 'assistant-content';
  content.append(paragraph, createResponseDetails(snapshot));
  const reactionFigure = createReactionFigure(snapshot.reaction, 'inline-reaction');
  article.replaceChildren(...(reactionFigure ? [reactionFigure, content] : [content]));

  const id = `weather-${++weatherSequence}`;
  article.dataset.weatherId = id;
  article.tabIndex = 0;
  article.setAttribute('role', 'button');
  article.setAttribute('aria-label', 'Show affective weather for this assistant response');
  article.setAttribute('aria-pressed', 'false');
  article.addEventListener('click', () => weatherHistory.select(id));
  article.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      weatherHistory.select(id);
    }
  });
  weatherHistory.add(id, snapshot);
  if (keepAtBottom) {
    const image = reactionFigure?.querySelector('img');
    if (image && !image.complete) image.addEventListener('load', () => scrollTranscriptToBottom(), { once: true });
    requestAnimationFrame(() => scrollTranscriptToBottom());
  }
}

async function reportFrontendRender(debugId, renderedText) {
  if (!debugId) return null;
  try {
    const response = await fetch(`/api/generation/${encodeURIComponent(debugId)}/rendered`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: renderedText }),
    });
    if (!response.ok) return null;
    const result = await response.json();
    return result.server_frontend_text_match;
  } catch {
    return null;
  }
}

liveWeatherButton.addEventListener('click', () => weatherHistory.returnToLive());

form.addEventListener('submit', async event => {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!text || button.disabled) return;
  const keepAtBottom = isNearTranscriptBottom();
  addMessage('user', text, '', keepAtBottom);
  messages.push({ role: 'user', content: text });
  prompt.value = '';
  button.disabled = true;
  const thinking = addMessage('assistant', 'Listening…', 'thinking', keepAtBottom);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, layer: Number(layerInput.value) }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Local inference failed');
    const manifest = await reactionManifestPromise;
    const reaction = selectMatchedReaction(manifest.assets, result.reaction_match?.candidates, { recent: recentReactionIds })
      || selectNearestReaction(manifest.assets, result.affect.valence, result.affect.arousal, {
        dominance: result.affect.dominance,
        recent: recentReactionIds,
      });
    if (!reaction.asset && manifest.loadError) reaction.fallbackReason = manifest.loadError;
    if (reaction.asset) {
      recentReactionIds.unshift(reaction.asset.id);
      recentReactionIds.splice(RECENT_REACTION_LIMIT);
    }
    thinking.remove();
    const article = addMessage('assistant', result.assistant_text, '', keepAtBottom);
    result.generation.server_frontend_text_match = await reportFrontendRender(
      result.generation.debug_id,
      article.querySelector('p').textContent,
    );
    const weatherSnapshot = { ...result, reaction, shape: readoutShape(result), manifestHealth: buildManifestHealth(manifest) };
    attachWeatherSnapshot(article, weatherSnapshot, keepAtBottom);
    messages.push({ role: 'assistant', content: result.assistant_text });
  } catch (error) {
    thinking.textContent = error.message;
    thinking.classList.remove('thinking');
  } finally {
    button.disabled = false;
    prompt.focus();
  }
});

prompt.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
