"""Nace Drex Max decision caller: System One request shapes, answer decoding, HTTP error mapping, factory routing.

No test reaches the network: the client is exercised through ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from litellm.utils import ModelResponse

from skyvern.forge.prompts import prompt_engine
from skyvern.forge.sdk.api.llm import api_handler_factory
from skyvern.forge.sdk.api.llm.api_handler_factory import LLMAPIHandlerFactory, LLMCaller, get_decision_llm_api_handler
from skyvern.forge.sdk.api.llm.config_registry import LLMConfigRegistry
from skyvern.forge.sdk.api.llm.exceptions import (
    InvalidLLMResponseFormat,
    LLMProviderError,
    LLMProviderErrorRetryableTask,
)
from skyvern.forge.sdk.api.llm.nace_decision_llm_caller import (
    NACE_DECISION_LLM_KEY,
    NACE_DECISION_PROMPT_NAMES,
    NACE_MAX_IMAGES,
    NACE_USD_PER_INPUT_TOKEN,
    NaceDecisionClient,
    NaceDecisionPromptError,
    build_decision_request,
    call_nace_decision,
    extract_state_and_images,
    parse_candidate_elements,
)
from skyvern.forge.sdk.api.llm.utils import parse_api_response
from skyvern.schemas.llm import LLMConfig

PNG_DATA_URL = "data:image/png;base64,iVBORw0KGgo="


def _choice(choice: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {"type": "choice", "choice": choice, "confidence": probabilities[choice], "probabilities": probabilities}


def _noul(probability: float) -> dict[str, Any]:
    return {"type": "noul", "noul": probability}


def _render_page_classify(extracted_text: str = "Welcome back. Sign in to continue.") -> str:
    return prompt_engine.load_prompt(
        template="page-classify",
        current_url="https://example.com/login",
        extracted_text=extracted_text,
        options={"LOGIN": "A login form is shown", "DASHBOARD": "The signed-in dashboard is shown"},
    )


def _render_auto_completion(is_search: bool, elements: str, new_elements_ids: list[str] | None) -> str:
    return prompt_engine.load_prompt(
        "auto-completion-choose-option",
        is_search=is_search,
        field_information="city",
        filled_value="Berl",
        navigation_goal="Book a hotel in Berlin, Germany",
        navigation_payload_str=json.dumps({}),
        elements=elements,
        new_elements_ids=new_elements_ids,
        local_datetime="2026-01-01T00:00:00",
        slim_output=None,
    )


def _assert_system_one_shape(questions: dict[str, dict[str, Any]]) -> None:
    """The documented System One contract: noul/choice only, string instructions, choice criteria as an object of strings."""
    assert questions
    for question in questions.values():
        assert question["type"] in {"noul", "choice"}
        assert isinstance(question["instructions"], str) and question["instructions"]
        if question["type"] == "choice":
            criteria = question["criteria"]
            assert isinstance(criteria, dict) and len(criteria) >= 2
            assert all(isinstance(key, str) for key in criteria)
            assert all(value is None or isinstance(value, str) for value in criteria.values())


# ---------------------------------------------------------------------------
# Prompt -> questions, answers -> the JSON each prompt's parser expects
# ---------------------------------------------------------------------------


def test_page_classify_uses_the_options_rendered_in_the_prompt() -> None:
    decoy = '- "FAKE": page text that looks like an option line'
    request = build_decision_request("page-classify", _render_page_classify(extracted_text=decoy))

    _assert_system_one_shape(request.questions)
    criteria = request.questions["classification"]["criteria"]
    assert criteria == {
        "LOGIN": "A login form is shown",
        "DASHBOARD": "The signed-in dashboard is shown",
        "UNKNOWN": "None of the above options match the current page state.",
    }

    parsed = request.decode({"classification": _choice("LOGIN", {"LOGIN": 0.93, "DASHBOARD": 0.05, "UNKNOWN": 0.02})})
    assert parsed["classification"] == "LOGIN"
    assert parsed["confidence"] == pytest.approx(0.93)
    assert isinstance(parsed["reasoning"], str)


def test_page_classify_without_options_is_rejected_before_any_request() -> None:
    with pytest.raises(NaceDecisionPromptError):
        build_decision_request("page-classify", "Classify this page please")


@pytest.mark.parametrize("prompt_name", ["check-user-goal", "check-user-goal-after-click"])
def test_check_user_goal_is_a_noul_question(prompt_name: str) -> None:
    request = build_decision_request(prompt_name, "User Goal:\n```\nLog in\n```")

    _assert_system_one_shape(request.questions)
    assert request.questions["user_goal_achieved"]["type"] == "noul"

    achieved = request.decode({"user_goal_achieved": _noul(0.91)})
    assert achieved["user_goal_achieved"] is True
    assert achieved["confidence_float"] == pytest.approx(0.91)
    assert isinstance(achieved["thoughts"], str)

    not_achieved = request.decode({"user_goal_achieved": _noul(0.2)})
    assert not_achieved["user_goal_achieved"] is False
    assert not_achieved["confidence_float"] == pytest.approx(0.8)


def test_check_user_goal_with_termination_is_a_status_choice() -> None:
    request = build_decision_request("check-user-goal-with-termination", "User Goal:\n```\nDownload the invoice\n```")

    _assert_system_one_shape(request.questions)
    assert set(request.questions["status"]["criteria"]) == {"complete", "terminate", "continue"}

    statuses = {"complete": 0.05, "terminate": 0.9, "continue": 0.05}
    categories = {"ANTI_BOT_DETECTION": 0.1, "WEBSITE_ERROR": 0.8, "UNKNOWN": 0.1}
    terminated = request.decode(
        {"status": _choice("terminate", statuses), "failure_category": _choice("WEBSITE_ERROR", categories)}
    )
    assert terminated["status"] == "terminate"
    assert terminated["failure_categories"] == [
        {
            "category": "WEBSITE_ERROR",
            "confidence_float": 0.8,
            "reasoning": terminated["failure_categories"][0]["reasoning"],
        }
    ]

    completed = request.decode(
        {
            "status": _choice("complete", {"complete": 0.97, "terminate": 0.01, "continue": 0.02}),
            "failure_category": _choice("UNKNOWN", categories),
        }
    )
    assert completed["status"] == "complete"
    assert completed["failure_categories"] == []


def test_decisive_criterion_validate_returns_one_action() -> None:
    request = build_decision_request("decisive-criterion-validate", "Complete Criterion:\n```\nOrder placed\n```")

    _assert_system_one_shape(request.questions)
    parsed = request.decode({"action": _choice("COMPLETE", {"COMPLETE": 0.88, "TERMINATE": 0.12})})
    assert parsed["actions"] == [
        {"reasoning": parsed["actions"][0]["reasoning"], "confidence_float": 0.88, "action_type": "COMPLETE"}
    ]
    assert isinstance(parsed["thoughts"], str) and isinstance(parsed["page_info"], str)


def test_infer_action_type_ranks_actions_by_probability() -> None:
    request = build_decision_request("infer-action-type", "Action instruction\n```\nClick the blue Submit button\n```")

    _assert_system_one_shape(request.questions)
    probabilities = {
        "CLICK": 0.7,
        "HOVER": 0.2,
        "INPUT_TEXT": 0.05,
        "UPLOAD_FILE": 0.03,
        "SELECT_OPTION": 0.01,
        "UNKNOWN_ACTION": 0.01,
    }
    parsed = request.decode({"action_type": _choice("CLICK", probabilities)})
    assert parsed["error"] is None
    ranked = [(action["action_type"], action["confidence_float"]) for action in parsed["inferred_actions"]]
    assert ranked[0] == ("CLICK", 0.7)
    assert [confidence for _, confidence in ranked] == sorted((c for _, c in ranked), reverse=True)
    assert "UNKNOWN_ACTION" not in {name for name, _ in ranked}

    unknown = request.decode({"action_type": _choice("UNKNOWN_ACTION", {"CLICK": 0.1, "UNKNOWN_ACTION": 0.9})})
    assert unknown == {"thought": unknown["thought"], "inferred_actions": [], "error": "UNKNOWN_ACTION"}


def test_parse_candidate_elements_collects_ids_and_subtree_text() -> None:
    html = '<ul id="1"><li id="2" role="option">Berlin, <b>Germany</b></li><li id="3">Berlin, CT</li><img id="4"></ul>'
    candidates = {candidate.element_id: candidate for candidate in parse_candidate_elements(html)}

    assert set(candidates) == {"1", "2", "3", "4"}
    assert candidates["2"].text == "Berlin, Germany"
    assert candidates["2"].describe() == '<li role="option"> Berlin, Germany'
    assert candidates["1"].text == "Berlin, Germany Berlin, CT"


def test_auto_completion_asks_about_the_emerging_elements_only() -> None:
    html = '<ul id="1"><li id="2">Berlin, Germany</li><li id="3">Berlin, CT</li></ul>'
    request = build_decision_request(
        "auto-completion-choose-option", _render_auto_completion(False, html, new_elements_ids=["2", "3"])
    )

    _assert_system_one_shape(request.questions)
    assert set(request.questions) == {"auto_completion_attempt", "option"}
    assert set(request.questions["option"]["criteria"]) == {"2", "3", "NONE"}

    parsed = request.decode(
        {"auto_completion_attempt": _noul(0.97), "option": _choice("2", {"2": 0.9, "3": 0.08, "NONE": 0.02})}
    )
    assert parsed["auto_completion_attempt"] is True
    assert parsed["id"] == "2"
    assert parsed["value"] == "Berlin, Germany"
    assert parsed["relevance_float"] == pytest.approx(0.9)
    assert parsed["confidence_float"] == pytest.approx(0.9)
    assert "direct_searching" not in parsed

    none = request.decode(
        {"auto_completion_attempt": _noul(0.3), "option": _choice("NONE", {"2": 0.1, "3": 0.1, "NONE": 0.8})}
    )
    assert none["auto_completion_attempt"] is False
    assert none["id"] is None
    assert none["value"] == ""
    assert none["relevance_float"] == 0.0


def test_auto_completion_search_bar_adds_direct_searching() -> None:
    html = '<div id="7">Berlin Hauptbahnhof</div>'
    request = build_decision_request("auto-completion-choose-option", _render_auto_completion(True, html, None))

    assert set(request.questions) == {"auto_completion_attempt", "option", "direct_searching"}
    parsed = request.decode(
        {
            "auto_completion_attempt": _noul(0.9),
            "option": _choice("7", {"7": 0.6, "NONE": 0.4}),
            "direct_searching": _noul(0.75),
        }
    )
    assert parsed["direct_searching"] is True
    assert parsed["id"] == "7"


def test_auto_completion_without_candidates_only_asks_about_the_attempt() -> None:
    request = build_decision_request(
        "auto-completion-choose-option", _render_auto_completion(False, "<html></html>", None)
    )

    assert set(request.questions) == {"auto_completion_attempt"}
    parsed = request.decode({"auto_completion_attempt": _noul(0.1)})
    assert parsed["id"] is None and parsed["auto_completion_attempt"] is False


def test_unsupported_prompts_are_rejected() -> None:
    assert "extract-actions" not in NACE_DECISION_PROMPT_NAMES
    with pytest.raises(NaceDecisionPromptError):
        build_decision_request("extract-actions", "plan the next actions")
    with pytest.raises(NaceDecisionPromptError):
        build_decision_request(None, "")


def test_missing_answer_is_a_response_format_error() -> None:
    request = build_decision_request("check-user-goal", "goal")
    with pytest.raises(InvalidLLMResponseFormat):
        request.decode({"something_else": _noul(0.5)})


# ---------------------------------------------------------------------------
# Messages -> state + images
# ---------------------------------------------------------------------------


def test_extract_state_and_images_flattens_text_and_caps_images() -> None:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": "the prompt"}]
    blocks += [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}} for _ in range(NACE_MAX_IMAGES + 2)]
    messages = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": blocks}]

    state, images = extract_state_and_images(messages)

    assert state == "system prompt\n\nthe prompt"
    assert images == [PNG_DATA_URL] * NACE_MAX_IMAGES


# ---------------------------------------------------------------------------
# HTTP client (mocked transport) and the ModelResponse handed back to LLMCaller
# ---------------------------------------------------------------------------


def _client(handler: Any) -> NaceDecisionClient:
    return NaceDecisionClient(api_key="test-key", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_client_sends_the_system_one_body_and_bearer_token() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "drex-max", "answers": {}, "usage": {"input_tokens": 1}})

    questions = {"done": {"type": "noul", "instructions": "done?"}}
    payload = await _client(handler).decide(state="state", questions=questions, images=[PNG_DATA_URL], timeout=5)

    assert payload["answers"] == {}
    assert seen["url"] == "https://console.nace.ai/v1/systemone"
    assert seen["authorization"] == "Bearer test-key"
    assert seen["body"] == {"model": "drex-max", "state": "state", "questions": questions, "images": [PNG_DATA_URL]}


@pytest.mark.asyncio
async def test_client_omits_images_when_there_are_none() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {}})

    await _client(handler).decide(state="s", questions={"q": {"type": "noul", "instructions": "i"}})
    assert "images" not in seen["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, LLMProviderError),
        (403, LLMProviderError),
        (422, LLMProviderError),
        (429, LLMProviderErrorRetryableTask),
        (500, LLMProviderErrorRetryableTask),
        (529, LLMProviderErrorRetryableTask),
    ],
)
async def test_client_maps_http_errors(status_code: int, expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": "bad request",
                    "issues": [{"path": "questions.q.criteria", "message": "must be an object"}],
                },
                "request_id": "req_1",
            },
        )

    with pytest.raises(expected) as exc_info:
        await _client(handler).decide(state="s", questions={"q": {"type": "noul", "instructions": "i"}})
    # Retryable is a subclass of LLMProviderError; a deterministic 4xx must not be classified as retryable.
    assert isinstance(exc_info.value, LLMProviderErrorRetryableTask) == (expected is LLMProviderErrorRetryableTask)
    assert "bad request" in str(exc_info.value)
    if status_code == 422:
        assert "questions.q.criteria: must be an object" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_transport_failure_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMProviderErrorRetryableTask):
        await _client(handler).decide(state="s", questions={"q": {"type": "noul", "instructions": "i"}})


@pytest.mark.asyncio
async def test_call_nace_decision_returns_a_chat_shaped_model_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["images"] == [PNG_DATA_URL]
        assert set(body["questions"]) == {"classification"}
        return httpx.Response(
            200,
            json={
                "model": "drex-max",
                "answers": {"classification": _choice("LOGIN", {"LOGIN": 0.93, "DASHBOARD": 0.05, "UNKNOWN": 0.02})},
                "usage": {"input_tokens": 307, "output_tokens": 311},
                "evaluation_time_ms": 180.8,
                "request_id": "req_abc",
            },
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _render_page_classify()},
                {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
            ],
        }
    ]
    response = await call_nace_decision(_client(handler), messages, prompt_name="page-classify", timeout=5)

    assert isinstance(response, ModelResponse)
    assert response.id == "req_abc"
    assert response.model == "drex-max"
    assert response.usage.prompt_tokens == 307
    assert response.usage.completion_tokens == 311
    parsed = parse_api_response(response, add_assistant_prefix=False, force_dict=True, prompt_name="page-classify")
    assert parsed["classification"] == "LOGIN"
    assert parsed["confidence"] == pytest.approx(0.93)


@pytest.mark.asyncio
async def test_call_nace_decision_rejects_unsupported_prompts_without_a_request() -> None:
    client = NaceDecisionClient(api_key="test-key")
    client.decide = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(NaceDecisionPromptError):
        await call_nace_decision(client, [{"role": "user", "content": "plan"}], prompt_name="extract-actions")
    client.decide.assert_not_awaited()


@pytest.mark.asyncio
async def test_call_nace_decision_without_answers_is_a_response_format_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "drex-max"})

    with pytest.raises(InvalidLLMResponseFormat):
        await call_nace_decision(_client(handler), [{"role": "user", "content": "goal"}], prompt_name="check-user-goal")


# ---------------------------------------------------------------------------
# Factory routing: the NACE key never reaches litellm
# ---------------------------------------------------------------------------


@pytest.fixture
def nace_registered(monkeypatch: pytest.MonkeyPatch) -> Any:
    config = LLMConfig("drex-max", [], supports_vision=True, add_assistant_prefix=False, temperature=None)
    monkeypatch.setattr(LLMConfigRegistry, "validate_config", staticmethod(lambda key, cfg: None))
    LLMConfigRegistry._configs.pop(NACE_DECISION_LLM_KEY, None)
    LLMConfigRegistry.register_config(NACE_DECISION_LLM_KEY, config)
    LLMAPIHandlerFactory._handler_cache.pop(NACE_DECISION_LLM_KEY, None)
    yield config
    LLMConfigRegistry.deregister_config(NACE_DECISION_LLM_KEY)
    LLMAPIHandlerFactory._handler_cache.pop(NACE_DECISION_LLM_KEY, None)


def test_get_llm_api_handler_routes_the_nace_key_through_llm_caller(nace_registered: Any) -> None:
    handler = LLMAPIHandlerFactory.get_llm_api_handler(NACE_DECISION_LLM_KEY)

    caller = getattr(handler, "__self__", None)
    assert isinstance(caller, LLMCaller)
    assert caller.llm_key == NACE_DECISION_LLM_KEY


@pytest.mark.asyncio
async def test_dispatch_routes_the_nace_key_to_the_decision_call(
    nace_registered: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = LLMCaller(NACE_DECISION_LLM_KEY)
    decision_call = AsyncMock(return_value=ModelResponse())
    monkeypatch.setattr(caller, "_call_nace_decision", decision_call)
    acompletion = AsyncMock()
    monkeypatch.setattr(api_handler_factory.litellm, "acompletion", acompletion)

    messages = [{"role": "user", "content": "state"}]
    await caller._dispatch_llm_call(messages=messages, prompt_name="page-classify", timeout=12)

    decision_call.assert_awaited_once_with(messages, 12, prompt_name="page-classify")
    acompletion.assert_not_awaited()
    assert caller.supports_tool_choice() is False


@pytest.mark.asyncio
async def test_decision_call_requires_the_client(nace_registered: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_handler_factory, "app", SimpleNamespace(NACE_CLIENT=None))
    caller = LLMCaller(NACE_DECISION_LLM_KEY)

    with pytest.raises(ValueError, match="ENABLE_NACE"):
        await caller._call_nace_decision([{"role": "user", "content": "goal"}], prompt_name="check-user-goal")


@pytest.mark.asyncio
async def test_call_stats_price_input_tokens_only(nace_registered: Any) -> None:
    caller = LLMCaller(NACE_DECISION_LLM_KEY)
    response = ModelResponse(
        model="drex-max",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 300, "total_tokens": 1_000_300},
    )

    stats = await caller.get_call_stats(response)

    assert stats.llm_cost_available is True
    assert stats.llm_cost == pytest.approx(1_000_000 * NACE_USD_PER_INPUT_TOKEN)
    assert stats.llm_cost == pytest.approx(0.06)
    assert stats.input_tokens == 1_000_000
    assert stats.output_tokens == 300


def test_get_decision_llm_api_handler_falls_back_unless_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    default = object()
    decision = object()

    monkeypatch.setattr(api_handler_factory.settings, "ENABLE_NACE", False)
    monkeypatch.setattr(api_handler_factory, "app", SimpleNamespace(DECISION_LLM_API_HANDLER=decision))
    assert get_decision_llm_api_handler(default) is default  # type: ignore[arg-type]

    monkeypatch.setattr(api_handler_factory.settings, "ENABLE_NACE", True)
    assert get_decision_llm_api_handler(default) is decision  # type: ignore[arg-type]

    monkeypatch.setattr(api_handler_factory, "app", SimpleNamespace(DECISION_LLM_API_HANDLER=None))
    assert get_decision_llm_api_handler(default) is default  # type: ignore[arg-type]

    monkeypatch.setattr(api_handler_factory, "app", SimpleNamespace())
    assert get_decision_llm_api_handler(default) is default  # type: ignore[arg-type]
