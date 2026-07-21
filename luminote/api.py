from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

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


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    layer: int | None = None


class FrontendRenderRequest(BaseModel):
    text: str


app = FastAPI(title="Luminote", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
app.mount("/assets", StaticFiles(directory=ROOT / "data"), name="assets")
app.mount("/reaction-assets", StaticFiles(directory=ROOT / "luminote-reactions" / "app-assets"), name="reaction-assets")
app.mount("/reaction-metadata", StaticFiles(directory=ROOT / "luminote-reactions" / "metadata"), name="reaction-metadata")
_model: LocalModel | None = None
_estimator: PrototypeEstimator | None = None
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


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {
        "model": settings.model_name,
        "calibrated": settings.prototype_path.exists(),
        "default_layer": settings.default_layer,
        "local_only": True,
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    model, estimator = services()
    layer = request.layer if request.layer is not None else settings.default_layer
    layer = max(0, min(layer, model.layer_count - 1))
    boundary = model.inspect_boundary(request.messages)
    estimate = estimator.estimate(boundary.by_layer[layer], layer)
    color = _colors.display_color(estimate.valence, estimate.arousal, estimate.confidence)
    generation = model.generate(request.messages, settings.max_new_tokens)
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
        "affect": estimate.to_dict(),
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
