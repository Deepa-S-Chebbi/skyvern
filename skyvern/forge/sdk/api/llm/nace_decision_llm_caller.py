"""Nace Drex Max decision caller for Skyvern's bounded-choice prompts.

Drex Max (https://console.nace.ai/docs/drex/guides/drex-max) is a decision model rather than a chat model: one
``POST /v1/systemone`` request carries a ``state`` text, up to eight images and a few bounded questions, and the
answer is a probability per option instead of generated text. Several Skyvern prompts are bounded questions
already (is the goal achieved? which page state? which action type? which suggestion?), so this module rewrites
each of them into System One questions and rebuilds the JSON object the prompt's existing parser expects. Nothing
here generates free text: ``thoughts``/``reasoning`` fields carry the decision and its probability.

Opt-in only. ``ENABLE_NACE=true`` registers the ``NACE_DREX_MAX`` LLM key, ``forge_app`` builds
``DECISION_LLM_API_HANDLER`` from it, and only the prompts in ``NACE_DECISION_PROMPT_NAMES`` are routed through
it (``get_decision_llm_api_handler`` in ``api_handler_factory``). Any other prompt sent to the key is rejected
before a request is made. Routing stays inside ``LLMCaller`` (``_call_nace_decision``), so artifact persistence,
token accounting and error classification are the base class's, as for UI-TARS and Yutori Navigator.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

import httpx
import structlog
from litellm.utils import ModelResponse

from skyvern.forge.sdk.api.llm.exceptions import (
    BaseLLMError,
    InvalidLLMResponseFormat,
    LLMProviderError,
    LLMProviderErrorRetryableTask,
)

LOG = structlog.get_logger()

NACE_DECISION_LLM_KEY = "NACE_DREX_MAX"
NACE_DEFAULT_API_BASE = "https://console.nace.ai"
NACE_DEFAULT_MODEL = "drex-max"
NACE_SYSTEMONE_PATH = "/v1/systemone"
# https://console.nace.ai/docs/drex/guides/drex-max: 8 images per request, 255 options per choice question.
NACE_MAX_IMAGES = 8
NACE_MAX_CHOICE_OPTIONS = 255
# Drex Max bills input tokens only ($0.06 per million); output tokens are free.
NACE_USD_PER_INPUT_TOKEN = 0.06 / 1_000_000

CHECK_USER_GOAL_PROMPT_NAME = "check-user-goal"
CHECK_USER_GOAL_AFTER_CLICK_PROMPT_NAME = "check-user-goal-after-click"
CHECK_USER_GOAL_WITH_TERMINATION_PROMPT_NAME = "check-user-goal-with-termination"
DECISIVE_CRITERION_VALIDATE_PROMPT_NAME = "decisive-criterion-validate"
PAGE_CLASSIFY_PROMPT_NAME = "page-classify"
INFER_ACTION_TYPE_PROMPT_NAME = "infer-action-type"
AUTO_COMPLETION_CHOOSE_OPTION_PROMPT_NAME = "auto-completion-choose-option"

# The only prompts this caller answers. Each one is a bounded decision whose JSON reply the recipe below rebuilds.
NACE_DECISION_PROMPT_NAMES: frozenset[str] = frozenset(
    {
        CHECK_USER_GOAL_PROMPT_NAME,
        CHECK_USER_GOAL_AFTER_CLICK_PROMPT_NAME,
        CHECK_USER_GOAL_WITH_TERMINATION_PROMPT_NAME,
        DECISIVE_CRITERION_VALIDATE_PROMPT_NAME,
        PAGE_CLASSIFY_PROMPT_NAME,
        INFER_ACTION_TYPE_PROMPT_NAME,
        AUTO_COMPLETION_CHOOSE_OPTION_PROMPT_NAME,
    }
)

_DECISION_NOTE_PREFIX = "Drex Max decision"


class NaceDecisionPromptError(BaseLLMError):
    """The prompt cannot be turned into a System One request (unsupported prompt or options not found)."""

    def __init__(self, prompt_name: str | None, detail: str) -> None:
        super().__init__(f"Nace Drex Max cannot answer prompt {prompt_name!r}: {detail}")
        self.prompt_name = prompt_name


class NaceDecisionAPIError(Exception):
    """A non-2xx reply from the System One API; wrapped in the LLMProviderError the caller raises."""

    def __init__(self, status_code: int, message: str, request_id: str | None = None) -> None:
        super().__init__(f"HTTP {status_code}: {message}" + (f" (request_id={request_id})" if request_id else ""))
        self.status_code = status_code
        self.request_id = request_id


def nace_decision_cost_usd(input_tokens: int) -> float:
    return input_tokens * NACE_USD_PER_INPUT_TOKEN


# ---------------------------------------------------------------------------
# System One questions and answers
# ---------------------------------------------------------------------------

Decoder = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class DecisionRequest:
    """The System One questions for one prompt and the decoder that rebuilds the prompt's JSON reply."""

    questions: dict[str, dict[str, Any]]
    decode: Decoder


def _noul(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


def _choice(instructions: str, criteria: Mapping[str, str | None]) -> dict[str, Any]:
    # criteria is an object keyed by option; values are plain strings (or null), never nested objects.
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def _noul_probability(answers: Mapping[str, Any], name: str) -> float:
    answer = answers.get(name)
    if not isinstance(answer, Mapping) or not isinstance(answer.get("noul"), (int, float)):
        raise InvalidLLMResponseFormat(f"missing noul answer {name!r}: {json.dumps(answers)[:500]}")
    return float(answer["noul"])


def _choice_answer(answers: Mapping[str, Any], name: str) -> tuple[str, float, dict[str, float]]:
    answer = answers.get(name)
    if not isinstance(answer, Mapping) or not isinstance(answer.get("choice"), str):
        raise InvalidLLMResponseFormat(f"missing choice answer {name!r}: {json.dumps(answers)[:500]}")
    raw_probabilities = answer.get("probabilities")
    probabilities = (
        {str(option): float(probability) for option, probability in raw_probabilities.items()}
        if isinstance(raw_probabilities, Mapping)
        else {}
    )
    choice = str(answer["choice"])
    confidence = answer.get("confidence")
    if not isinstance(confidence, (int, float)):
        confidence = probabilities.get(choice, 0.0)
    return choice, float(confidence), probabilities


def _binary_confidence(probability: float) -> float:
    return max(probability, 1.0 - probability)


def _note(what: str, probability: float) -> str:
    return f"{_DECISION_NOTE_PREFIX}: {what} (p={probability:.3f})"


def _fenced_block_after(prompt: str, heading: str) -> str | None:
    """The contents of the ``` fenced block right after the line ``heading`` in a rendered prompt, or None.

    Anchored to a whole line: the templates also mention these headings inside their reply-format comments.
    """
    match = re.search(rf"^{re.escape(heading)}\n```[a-z]*\n(.*?)\n```", prompt, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else None


# -- check-user-goal / check-user-goal-after-click ---------------------------------------------------------------

_CHECK_USER_GOAL_INSTRUCTIONS = (
    "The text is a Skyvern verification prompt: a user goal, user details, the action history and the elements "
    "parsed from the current page; any images are screenshots of that page. Judge it the way the prompt instructs. "
    "Has the user goal been completed?"
)


def _check_user_goal_request(prompt: str) -> DecisionRequest:
    questions = {"user_goal_achieved": _noul(_CHECK_USER_GOAL_INSTRUCTIONS)}

    def decode(answers: dict[str, Any]) -> dict[str, Any]:
        probability = _noul_probability(answers, "user_goal_achieved")
        achieved = probability >= 0.5
        return {
            "thoughts": _note("user goal achieved" if achieved else "user goal not achieved", probability),
            "user_goal_achieved": achieved,
            "confidence_float": _binary_confidence(probability),
        }

    return DecisionRequest(questions=questions, decode=decode)


# -- check-user-goal-with-termination ----------------------------------------------------------------------------

_VERIFICATION_STATUSES: dict[str, str] = {
    "complete": "the user goal has been fully achieved (according to the complete criterion when one is given)",
    "terminate": (
        "the goal can never be achieved: a permanent error, a missing file, or an explicit termination condition "
        "from the user goal is met"
    ),
    "continue": (
        "the goal is not achieved yet but more steps could achieve it: the page is loading, something must be "
        "clicked, or a different action can be tried"
    ),
}

_FAILURE_CATEGORIES: dict[str, str] = {
    "ANTI_BOT_DETECTION": "CAPTCHA, Cloudflare challenge or bot detection",
    "BROWSER_ERROR": "browser crash or page closed",
    "NAVIGATION_FAILURE": "404 or redirect",
    "PAGE_LOAD_TIMEOUT": "page load timeout",
    "AUTH_FAILURE": "login failed or MFA",
    "LLM_REASONING_ERROR": "wrong element or wrong action",
    "CREDENTIAL_ERROR": "missing credentials",
    "ELEMENT_NOT_FOUND": "target element missing",
    "WRONG_PAGE_STATE": "unexpected page",
    "DATA_EXTRACTION_FAILURE": "extraction failed",
    "WEBSITE_ERROR": "the website's own server failed or is down (HTTP 5xx, CDN error, outage or maintenance page)",
    "INFRASTRUCTURE_ERROR": "Skyvern's own platform failed (workers, browser pool, proxies, network, memory)",
    "UNKNOWN": "the root cause cannot be determined",
}


def _check_user_goal_with_termination_request(prompt: str) -> DecisionRequest:
    questions = {
        "status": _choice(
            "The text is a Skyvern verification prompt: a user goal, user details, the action history and the "
            "elements parsed from the current page; any images are screenshots of that page. Judge it the way the "
            "prompt instructs. Which status describes the task?",
            _VERIFICATION_STATUSES,
        ),
        "failure_category": _choice(
            "If the goal can never be achieved, which category best explains the root cause?",
            _FAILURE_CATEGORIES,
        ),
    }

    def decode(answers: dict[str, Any]) -> dict[str, Any]:
        status, confidence, _ = _choice_answer(answers, "status")
        if status not in _VERIFICATION_STATUSES:
            raise InvalidLLMResponseFormat(f"unexpected verification status {status!r}")
        failure_categories: list[dict[str, Any]] = []
        if status == "terminate":
            category, category_confidence, _ = _choice_answer(answers, "failure_category")
            failure_categories.append(
                {
                    "category": category,
                    "confidence_float": category_confidence,
                    "reasoning": _note(f"failure category {category}", category_confidence),
                }
            )
        return {
            "thoughts": _note(f"status {status}", confidence),
            "status": status,
            "failure_categories": failure_categories,
            "confidence_float": confidence,
        }

    return DecisionRequest(questions=questions, decode=decode)


# -- decisive-criterion-validate ---------------------------------------------------------------------------------

_DECISIVE_ACTIONS: dict[str, str] = {
    "COMPLETE": "the complete criterion is met",
    "TERMINATE": "the terminate criterion is met, or a complete criterion was provided but is not met",
}


def _decisive_criterion_validate_request(prompt: str) -> DecisionRequest:
    questions = {
        "action": _choice(
            "The text is a Skyvern validation prompt: a complete and/or terminate criterion, user details and, "
            "unless it says the page is not relevant, the current URL and the elements parsed from the page; any "
            "images are screenshots of that page. Judge it the way the prompt instructs. Which action applies?",
            _DECISIVE_ACTIONS,
        )
    }

    def decode(answers: dict[str, Any]) -> dict[str, Any]:
        action, confidence, _ = _choice_answer(answers, "action")
        if action not in _DECISIVE_ACTIONS:
            raise InvalidLLMResponseFormat(f"unexpected validation action {action!r}")
        note = _note(f"action {action}", confidence)
        return {
            "page_info": note,
            "thoughts": note,
            "actions": [{"reasoning": note, "confidence_float": confidence, "action_type": action}],
        }

    return DecisionRequest(questions=questions, decode=decode)


# -- page-classify -----------------------------------------------------------------------------------------------

_PAGE_CLASSIFY_OPTIONS_START = "Classify the page into ONE of the following options:"
_PAGE_CLASSIFY_OPTIONS_END = "Reply in JSON format:"
_PAGE_CLASSIFY_OPTION = re.compile(r'^- "(?P<key>[^"]+)": (?P<description>.*)$', re.MULTILINE)


def parse_page_classify_options(prompt: str) -> dict[str, str]:
    """The option keys and descriptions rendered by page-classify.j2 (including UNKNOWN)."""
    start = prompt.find(_PAGE_CLASSIFY_OPTIONS_START)
    if start < 0:
        return {}
    end = prompt.find(_PAGE_CLASSIFY_OPTIONS_END, start)
    section = prompt[start : end if end > start else len(prompt)]
    return {match["key"]: match["description"].strip() for match in _PAGE_CLASSIFY_OPTION.finditer(section)}


def _page_classify_request(prompt: str) -> DecisionRequest:
    options = parse_page_classify_options(prompt)
    if not options:
        raise NaceDecisionPromptError(PAGE_CLASSIFY_PROMPT_NAME, "no classification options found in the prompt")
    questions = {
        "classification": _choice(
            "The text gives the current URL and the first characters of the page text. Which option best "
            "describes the current state of the page?",
            options,
        )
    }

    def decode(answers: dict[str, Any]) -> dict[str, Any]:
        classification, confidence, _ = _choice_answer(answers, "classification")
        return {
            "reasoning": _note(f"classification {classification}", confidence),
            "classification": classification,
            "confidence": confidence,
        }

    return DecisionRequest(questions=questions, decode=decode)


# -- infer-action-type -------------------------------------------------------------------------------------------

_UNKNOWN_ACTION = "UNKNOWN_ACTION"
_ACTION_TYPES: dict[str, str] = {
    "CLICK": "the user wants to click",
    "INPUT_TEXT": "the user wants to input text",
    "UPLOAD_FILE": "the user wants to upload a file",
    "SELECT_OPTION": "the user wants to select an option",
    "HOVER": (
        "the user wants to move the mouse over an element without clicking, such as to reveal a hover-only menu "
        "or tooltip"
    ),
    _UNKNOWN_ACTION: "none of the defined action types matches the instruction",
}


def _infer_action_type_request(prompt: str) -> DecisionRequest:
    questions = {
        "action_type": _choice(
            "The text is a single browser action instruction. Which action type does it ask for?",
            _ACTION_TYPES,
        )
    }

    def decode(answers: dict[str, Any]) -> dict[str, Any]:
        action_type, confidence, probabilities = _choice_answer(answers, "action_type")
        if action_type == _UNKNOWN_ACTION:
            return {
                "thought": _note("no defined action type matches the instruction", confidence),
                "inferred_actions": [],
                "error": _UNKNOWN_ACTION,
            }
        if action_type not in _ACTION_TYPES:
            raise InvalidLLMResponseFormat(f"unexpected action type {action_type!r}")
        ranked = sorted(
            ((name, probability) for name, probability in probabilities.items() if name in _ACTION_TYPES),
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked or ranked[0][0] != action_type:
            ranked = [(action_type, confidence)]
        return {
            "thought": _note(f"action type {action_type}", confidence),
            "inferred_actions": [
                {
                    "reasoning": _note(f"action type {name}", probability),
                    "confidence_float": probability,
                    "action_type": name,
                }
                for name, probability in ranked
                if name != _UNKNOWN_ACTION
            ],
            "error": None,
        }

    return DecisionRequest(questions=questions, decode=decode)


# -- auto-completion-choose-option -------------------------------------------------------------------------------

_AUTO_COMPLETION_SEARCH_MARKER = "or directly search with the input value"
_AUTO_COMPLETION_ELEMENTS_HEADING = "HTML elements:"
_AUTO_COMPLETION_EMERGING_HEADING = "IDs for emerging HTML elements"
_NO_OPTION = "NONE"
_MAX_CANDIDATE_DESCRIPTION_CHARS = 240
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
)


@dataclass
class CandidateElement:
    """An element with a Skyvern id rendered in a prompt, as a choice option."""

    element_id: str
    tag: str
    attributes: dict[str, str]
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self.text_parts).split())

    def describe(self) -> str:
        attributes = " ".join(f'{key}="{value}"' for key, value in self.attributes.items())
        description = f"<{self.tag}{' ' + attributes if attributes else ''}> {self.text}".strip()
        return description[:_MAX_CANDIDATE_DESCRIPTION_CHARS]


class _CandidateElementParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[CandidateElement] = []
        self._open: list[CandidateElement | None] = []

    def _start(self, tag: str, attrs: list[tuple[str, str | None]]) -> CandidateElement | None:
        attributes = {key: value for key, value in attrs if value}
        element_id = attributes.pop("id", None)
        if not element_id:
            return None
        candidate = CandidateElement(element_id=element_id, tag=tag, attributes=attributes)
        self.candidates.append(candidate)
        return candidate

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        candidate = self._start(tag, attrs)
        if tag not in _VOID_TAGS:
            self._open.append(candidate)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._open:
            self._open.pop()

    def handle_data(self, data: str) -> None:
        for candidate in self._open:
            if candidate is not None:
                candidate.text_parts.append(data)


def parse_candidate_elements(html: str) -> list[CandidateElement]:
    """Every element with an ``id`` attribute in a rendered element tree, with the text of its subtree."""
    parser = _CandidateElementParser()
    parser.feed(html)
    parser.close()
    return parser.candidates


def _auto_completion_request(prompt: str) -> DecisionRequest:
    is_search = _AUTO_COMPLETION_SEARCH_MARKER in prompt
    html = _fenced_block_after(prompt, _AUTO_COMPLETION_ELEMENTS_HEADING) or ""
    candidates = parse_candidate_elements(html)
    emerging = _fenced_block_after(prompt, _AUTO_COMPLETION_EMERGING_HEADING)
    if emerging:
        # The template renders the ids as a Python list/set literal, so each id appears quoted.
        emerging_candidates = [
            c for c in candidates if f"'{c.element_id}'" in emerging or f'"{c.element_id}"' in emerging
        ]
        if emerging_candidates:
            candidates = emerging_candidates
    if len(candidates) > NACE_MAX_CHOICE_OPTIONS - 1:
        LOG.info(
            "Drex Max choice question capped; dropping trailing auto-completion candidates",
            candidates=len(candidates),
            kept=NACE_MAX_CHOICE_OPTIONS - 1,
        )
        candidates = candidates[: NACE_MAX_CHOICE_OPTIONS - 1]
    values = {candidate.element_id: candidate.text for candidate in candidates}

    questions: dict[str, dict[str, Any]] = {
        "auto_completion_attempt": _noul(
            "The text shows an input field, the value typed into it, the user goal and details, and the elements "
            "that are on the page now. Did the page attempt auto-completion for the typed value? Several "
            "suggestions count, and so does a message like 'No results' or 'No match'."
        )
    }
    if candidates:
        criteria: dict[str, str | None] = {candidate.element_id: candidate.describe() for candidate in candidates}
        criteria[_NO_OPTION] = "no element is a meaningful auto-completion suggestion for the typed value"
        questions["option"] = _choice(
            "Each option is an element on the page, keyed by its id. Which element is the auto-completion "
            f"suggestion that best matches the user goal, the user details and the field context? Choose {_NO_OPTION} "
            "when no element is a meaningful suggestion (a 'No results' message is not one).",
            criteria,
        )
    if is_search:
        questions["direct_searching"] = _noul(
            "Is searching directly with the typed input value a better way to reach the user goal than selecting "
            "any of the suggestions?"
        )

    def decode(answers: dict[str, Any]) -> dict[str, Any]:
        attempt_probability = _noul_probability(answers, "auto_completion_attempt")
        element_id: str | None = None
        confidence = _binary_confidence(attempt_probability)
        relevance = 0.0
        if "option" in questions:
            option, confidence, probabilities = _choice_answer(answers, "option")
            if option != _NO_OPTION:
                if option not in values:
                    raise InvalidLLMResponseFormat(f"unexpected auto-completion option {option!r}")
                element_id = option
                relevance = round(probabilities.get(option, confidence), 2)
        result: dict[str, Any] = {
            "thought": _note(
                "auto-completion attempt" if attempt_probability >= 0.5 else "no attempt", attempt_probability
            ),
            "auto_completion_attempt": attempt_probability >= 0.5,
            "reasoning": _note(f"option {element_id}" if element_id else "no suitable option", confidence),
            "confidence_float": confidence,
            "relevance_float": relevance,
            "value": values.get(element_id, "") if element_id else "",
            "id": element_id,
        }
        if is_search:
            direct_probability = _noul_probability(answers, "direct_searching")
            result["direct_searching"] = direct_probability >= 0.5
        return result

    return DecisionRequest(questions=questions, decode=decode)


_RECIPES: dict[str, Callable[[str], DecisionRequest]] = {
    CHECK_USER_GOAL_PROMPT_NAME: _check_user_goal_request,
    CHECK_USER_GOAL_AFTER_CLICK_PROMPT_NAME: _check_user_goal_request,
    CHECK_USER_GOAL_WITH_TERMINATION_PROMPT_NAME: _check_user_goal_with_termination_request,
    DECISIVE_CRITERION_VALIDATE_PROMPT_NAME: _decisive_criterion_validate_request,
    PAGE_CLASSIFY_PROMPT_NAME: _page_classify_request,
    INFER_ACTION_TYPE_PROMPT_NAME: _infer_action_type_request,
    AUTO_COMPLETION_CHOOSE_OPTION_PROMPT_NAME: _auto_completion_request,
}


def build_decision_request(prompt_name: str | None, prompt: str) -> DecisionRequest:
    """The System One questions for ``prompt_name`` rendered as ``prompt``; rejects prompts this caller cannot answer."""
    recipe = _RECIPES.get(prompt_name or "")
    if recipe is None:
        raise NaceDecisionPromptError(
            prompt_name, f"only bounded-choice prompts are supported: {', '.join(sorted(NACE_DECISION_PROMPT_NAMES))}"
        )
    return recipe(prompt)


# ---------------------------------------------------------------------------
# Messages -> request, API call, answers -> ModelResponse
# ---------------------------------------------------------------------------


def extract_state_and_images(messages: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Flatten chat messages into the System One ``state`` text and the image data URLs (first 8 kept)."""
    texts: list[str] = []
    images: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for block in content or []:
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            if block_type == "text" and block.get("text"):
                texts.append(str(block["text"]))
            elif block_type == "image_url":
                url = block.get("image_url", {}).get("url")
                if url:
                    images.append(str(url))
            elif block_type == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64" and source.get("data"):
                    images.append(f"data:{source.get('media_type', 'image/png')};base64,{source['data']}")
    if len(images) > NACE_MAX_IMAGES:
        LOG.info("Drex Max accepts 8 images per request; dropping the rest", image_count=len(images))
        images = images[:NACE_MAX_IMAGES]
    return "\n\n".join(texts), images


class NaceDecisionClient:
    """Minimal async client for ``POST {base_url}/v1/systemone``.

    4xx replies are deterministic (bad key, invalid request) and surface as ``LLMProviderError``; 429, 529 and 5xx
    are transient and surface as ``LLMProviderErrorRetryableTask``, like the litellm paths do.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str = NACE_DEFAULT_API_BASE,
        model: str = NACE_DEFAULT_MODEL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "skyvern"},
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def decide(
        self,
        *,
        state: str,
        questions: dict[str, dict[str, Any]],
        images: list[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if images:
            body["images"] = images
        try:
            response = await self._http().post(NACE_SYSTEMONE_PATH, json=body, timeout=timeout)
        except httpx.HTTPError as e:
            raise LLMProviderErrorRetryableTask(NACE_DECISION_LLM_KEY, cause=e) from e
        if response.status_code >= 400:
            error = NaceDecisionAPIError(
                response.status_code, _error_message(response), response.headers.get("x-request-id")
            )
            if response.status_code in (408, 429, 529) or response.status_code >= 500:
                raise LLMProviderErrorRetryableTask(NACE_DECISION_LLM_KEY, cause=error) from error
            raise LLMProviderError(NACE_DECISION_LLM_KEY, cause=error) from error
        try:
            payload = response.json()
        except ValueError as e:
            raise LLMProviderErrorRetryableTask(NACE_DECISION_LLM_KEY, cause=e) from e
        if not isinstance(payload, dict):
            raise InvalidLLMResponseFormat(f"System One reply is not an object: {str(payload)[:200]}")
        return payload


def _error_message(response: httpx.Response) -> str:
    try:
        error = response.json().get("error") or {}
    except ValueError:
        return response.text[:300]
    message = str(error.get("message") or response.text[:300])
    issues = error.get("issues") or []
    if issues:
        message += " " + "; ".join(f"{issue.get('path')}: {issue.get('message')}" for issue in issues[:5])
    return message


def _to_model_response(
    parsed: dict[str, Any], usage: Mapping[str, Any], model: str, request_id: str | None
) -> ModelResponse:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return ModelResponse(
        id=request_id or f"nace-{uuid.uuid4().hex}",
        model=model,
        object="chat.completion",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(parsed)},
            }
        ],
        usage={
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


async def call_nace_decision(
    client: NaceDecisionClient,
    messages: list[dict[str, Any]],
    prompt_name: str | None,
    timeout: float | None = None,
) -> ModelResponse:
    """Answer one bounded-choice prompt with Drex Max and return it as a chat-shaped ModelResponse.

    The content is the JSON object the prompt's parser expects, so ``parse_api_response`` and the call sites stay
    unchanged; ``usage`` carries the System One token counts for cost tracking.
    """
    state, images = extract_state_and_images(messages)
    request = build_decision_request(prompt_name, state)
    LOG.info(
        "Drex Max decision request",
        prompt_name=prompt_name,
        model=client.model,
        questions=list(request.questions),
        image_count=len(images),
    )
    payload = await client.decide(state=state, questions=request.questions, images=images, timeout=timeout)
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise InvalidLLMResponseFormat(f"System One reply has no answers: {json.dumps(payload)[:300]}")
    parsed = request.decode(answers)
    raw_usage = payload.get("usage")
    usage: Mapping[str, Any] = raw_usage if isinstance(raw_usage, Mapping) else {}
    LOG.info(
        "Drex Max decision response",
        prompt_name=prompt_name,
        request_id=payload.get("request_id"),
        evaluation_time_ms=payload.get("evaluation_time_ms"),
        input_tokens=usage.get("input_tokens"),
    )
    return _to_model_response(
        parsed, usage, model=str(payload.get("model") or client.model), request_id=payload.get("request_id")
    )
