from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .affect import PrototypeEstimator
from .color_map import ColorMap
from .config import ROOT, settings
from .generation_debug import GenerationIntegrityStore
from .logging import JsonlLogger
from .model import LocalModel
from .reaction_matching import ReactionIndex


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    layer: int | None = None
    # Kept separate from `layer`: the best layer for span readings differs from the boundary's.
    span_layer: int | None = None


class FrontendRenderRequest(BaseModel):
    text: str


app = FastAPI(title="Luminote", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
app.mount("/assets", StaticFiles(directory=ROOT / "data"), name="assets")
app.mount("/reaction-assets", StaticFiles(directory=ROOT / "luminote-reactions" / "app-assets"), name="reaction-assets")
app.mount("/reaction-metadata", StaticFiles(directory=ROOT / "luminote-reactions" / "metadata"), name="reaction-metadata")
_model: LocalModel | None = None
_estimator: PrototypeEstimator | None = None
# Built by scripts/place_reactions.py; restart the server after re-running it.
_reactions = ReactionIndex.load(settings.reaction_index_path)
_lock = Lock()
_colors = ColorMap(settings.color_map_path)
_logger = JsonlLogger(settings.log_path)
_generation_debug = GenerationIntegrityStore(JsonlLogger(settings.generation_log_path))


def services() -> tuple[LocalModel, PrototypeEstimator]:
    global _model, _estimator
    with _lock:
        if _model is None:
            _model = LocalModel(settings.model_name)
        if _estimator is None:
            if not settings.prototype_path.exists():
                raise HTTPException(503, "Calibration missing. Run ./scripts/calibrate.sh first.")
            _estimator = PrototypeEstimator(settings.prototype_path, settings.prototype_meta_path)
            if _estimator.model_name != _model.model_name or _estimator.layer_count != _model.layer_count:
                _estimator = None
                raise HTTPException(503, "Calibration does not match the active model. Re-run ./scripts/calibrate.sh.")
    return _model, _estimator


def read_spans(
    model: LocalModel,
    estimator: PrototypeEstimator,
    messages: list[dict[str, str]],
    reply: str,
    layer: int,
) -> tuple[dict, torch.Tensor | None]:
    """Per-token readings of the last user message and the reply as it was actually written.

    Returns the JSON payload and the reply's [token, hidden] activations for weather colouring.
    """
    if not reply.strip():
        return {"layer": layer, "error": "Empty reply."}, None
    try:
        activations = model.inspect_spans([*messages, {"role": "assistant", "content": reply}], [layer])
    except ValueError as error:
        return {"layer": layer, "error": str(error)}, None
    hidden = activations.layer(layer)
    reply_hidden = hidden[slice(*activations.reply_span)]
    result = {
        "layer": layer,
        "example_prototypes": estimator.span is not None,
        "reply": estimator.estimate_span(reply_hidden, layer).to_dict(),
        "reply_tokens": activations.tokens[slice(*activations.reply_span)],
        # The boundary token read with the same prototypes as the reply, so the two compare directly.
        "boundary": estimator.estimate_pooled(
            hidden[activations.boundary_index : activations.boundary_index + 1], layer
        ).to_dict(),
        "user": None,
    }
    if activations.user_span is not None:
        result["user"] = estimator.estimate_span(hidden[slice(*activations.user_span)], layer).to_dict()
    return result, reply_hidden


def match_reactions(
    model: LocalModel, estimator: PrototypeEstimator, reply_hidden: torch.Tensor | None, layer: int
) -> dict:
    """Rank reaction images by similarity to the reply in whitened activation space."""
    if _reactions is None:
        return {"candidates": [], "reason": "No reaction embeddings; run scripts/place_reactions.py."}
    if _reactions.model_name != model.model_name or _reactions.layer != layer:
        return {"candidates": [], "reason": "Reaction embeddings were built for another model or layer."}
    embedding = estimator.match_embedding(reply_hidden, layer) if reply_hidden is not None else None
    if embedding is None:
        return {"candidates": [], "reason": "Calibration has no reaction-matching whitener for this layer."}
    return {"layer": layer, "candidates": _reactions.rank(embedding), "reason": None}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {
        "model": settings.model_name,
        "calibrated": settings.prototype_path.exists(),
        "default_layer": settings.default_layer,
        "span_layer": settings.span_layer,
        "weather_source": settings.weather_source,
        "local_only": True,
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    model, estimator = services()
    layer = request.layer if request.layer is not None else settings.default_layer
    layer = max(0, min(layer, model.layer_count - 1))
    boundary = model.inspect_boundary(request.messages)
    estimate = estimator.estimate(boundary.by_layer[layer], layer)
    generation = model.generate(request.messages, settings.max_new_tokens)
    span_layer = request.span_layer if request.span_layer is not None else settings.span_layer
    span_layer = max(0, min(span_layer, model.layer_count - 1))
    spans, reply_hidden = read_spans(model, estimator, request.messages, generation.text, span_layer)
    reaction_match = match_reactions(model, estimator, reply_hidden, span_layer)
    weather = estimate
    if settings.weather_source == "reply" and reply_hidden is not None:
        weather = estimator.estimate_pooled(reply_hidden, span_layer)
    color = _colors.display_color(weather.valence, weather.arousal, weather.confidence)
    debug_id = uuid4().hex
    _generation_debug.record_generation(debug_id, {
        "input_messages": request.messages,
        "rendered_chat_template_prompt": generation.prompt_text,
        "generation_parameters": generation.parameters,
        "generated_token_ids": generation.token_ids,
        "server_decoded_text": generation.text,
        "generated_token_count": len(generation.token_ids),
        "stop_reason": generation.stop_reason,
    })
    payload = {
        "assistant_text": generation.text,
        "affect": weather.to_dict(),
        "boundary_affect": estimate.to_dict(),
        "spans": spans,
        "reaction_match": reaction_match,
        "weather_source": "reply" if weather is not estimate else "boundary",
        "color": color,
        "layer": layer,
        "layer_count": model.layer_count,
        "response_start_token": boundary.token_text,
        "response_start_token_id": boundary.token_id,
        "model": model.model_name,
        "generation": {
            "debug_id": debug_id,
            "stop_reason": generation.stop_reason,
            "generated_token_count": len(generation.token_ids),
            "server_frontend_text_match": None,
        },
    }
    _logger.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_text": request.messages[-1]["content"],
        **payload,
    })
    return payload


@app.post("/api/generation/{debug_id}/rendered")
def record_frontend_render(debug_id: str, request: FrontendRenderRequest) -> dict:
    matches = _generation_debug.record_frontend_text(debug_id, request.text)
    if matches is None:
        raise HTTPException(404, "Unknown generation debug ID.")
    return {"debug_id": debug_id, "server_frontend_text_match": matches}
