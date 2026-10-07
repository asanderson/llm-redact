"""Span decoding for BIOES and BILOU token taggers (the ``hf`` backend).

A token-classification model labels each token. BIO models mark a span's first
token ``B-`` and the rest ``I-``, which the transformers pipeline's aggregation
reads. BIOES (``B-``, ``I-``, ``E-``nd, ``S-``ingle, ``O``) and BILOU (``L-``ast
for E, ``U-``nit for S) models also mark a span's last token, and the pipeline
does not understand those tags: it would cut every span at its ``E-`` token and
merge single-token spans with their neighbours. Such models are decoded here,
from the model's per-token log-probabilities:

* :func:`viterbi_path` — a constrained Viterbi decoder: the best label path in
  which every span opens with ``B``/``S``, continues with ``I`` of the same
  entity and closes with ``E``/``S`` (a linear-chain model whose transition
  scores are six biases a model ships in its calibration file,
  :func:`viterbi_biases`; the transitions the scheme forbids score minus
  infinity). OpenAI's reference decoder for openai/privacy-filter is the
  model for it.
* :func:`spans_of` — the greedy reading of a label sequence (the argmax per
  token, or a Viterbi path): ``B`` … ``E`` and single ``S`` spans, ``I``
  continuing; a tag that cannot continue the open span (an ``I``/``E`` of
  another entity, or with no span open) starts a new one, and a span still
  open when an ``O`` or the text's end comes is kept as it is — a malformed
  sequence never drops a token the model marked.

Labels are read by position: the model's ``id2label`` must name every logit
index. Nothing here sees text or logs; it works on label indices and scores.
"""

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

# The tagging schemes (model_catalog.TAGGING_SCHEMES).
BIO = "bio"
BIOES = "bioes"
BILOU = "bilou"

# The calibration file's six transition biases (OpenAI's
# viterbi_calibration.json, operating point "default").
BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)

# One leading tag before a letter, as labels.normalize_label drops it.
_TAG_RE = re.compile(r"([BIESLUbieslu])-(?=[^\W\d_])")
# BILOU spelled as BIOES: L(ast) is E(nd), U(nit) is S(ingle).
_AS_BIOES = {"L": "E", "U": "S"}
_BACKGROUND = "O"
_NO_SCORE = -math.inf


class TaggingError(ValueError):
    """A model's labels or calibration cannot be decoded. The message names
    tags, keys and positions only."""


def tagging_scheme(labels: Iterable[str]) -> str:
    """The tagging scheme a model's labels use: :data:`BIOES` when any label
    carries an ``E-`` or ``S-`` tag, :data:`BILOU` for ``L-`` or ``U-``,
    else :data:`BIO`. Labels mixing both are refused."""
    tags = {match[1].upper() for label in labels if (match := _TAG_RE.match(label.strip()))}
    if tags & {"E", "S"} and tags & {"L", "U"}:
        raise TaggingError("its labels mix BIOES (E-, S-) and BILOU (L-, U-) tags")
    if tags & {"E", "S"}:
        return BIOES
    if tags & {"L", "U"}:
        return BILOU
    return BIO


@dataclass(frozen=True)
class TagSet:
    """What each logit index of a BIOES or BILOU model stands for: a tag —
    ``B``, ``I``, ``E``, ``S`` (BILOU's ``L``/``U`` read as ``E``/``S``),
    ``O`` for the background label, or None for a label without a tag, which
    no span uses — and the entity label after the tag ("" for the rest)."""

    tags: tuple[str | None, ...]
    entities: tuple[str, ...]

    @classmethod
    def from_labels(cls, id2label: Mapping[int, str] | Mapping[str, str]) -> "TagSet":
        """The tag set of a model's ``config.id2label`` (keys are the logit
        indices, as integers or their decimal strings, 0 to n-1)."""
        try:
            by_index = {int(index): str(label) for index, label in id2label.items()}
        except (TypeError, ValueError) as exc:
            raise TaggingError("its id2label keys are not logit indices") from exc
        if sorted(by_index) != list(range(len(by_index))):
            raise TaggingError("its id2label does not name every logit index 0 to n-1")
        tags: list[str | None] = []
        entities: list[str] = []
        for index in range(len(by_index)):
            label = by_index[index].strip()
            match = _TAG_RE.match(label)
            if match is not None:
                tag = match[1].upper()
                tags.append(_AS_BIOES.get(tag, tag))
                entities.append(label[2:])
            else:
                tags.append(_BACKGROUND if label.upper() == _BACKGROUND else None)
                entities.append("")
        return cls(tuple(tags), tuple(entities))

    @classmethod
    def from_bio_labels(cls, id2label: Mapping[int, str] | Mapping[str, str]) -> "TagSet":
        """The tag set of a BIO model read as the transformers pipeline
        reads one: a label without a tag ("PER") is ``I`` of itself."""
        tagged = cls.from_labels(id2label)
        labels = {int(index): str(label).strip() for index, label in id2label.items()}
        return cls(
            tuple("I" if tag is None else tag for tag in tagged.tags),
            tuple(
                labels[index] if tag is None else entity
                for index, (tag, entity) in enumerate(
                    zip(tagged.tags, tagged.entities, strict=True)
                )
            ),
        )

    def __len__(self) -> int:
        return len(self.tags)


def viterbi_biases(calibration: Mapping[str, object]) -> dict[str, float]:
    """The six transition biases of a calibration file's default operating
    point: ``{"operating_points": {"default": {"biases": {...}}}}`` with
    exactly the :data:`BIAS_KEYS`, each a number."""
    points = calibration.get("operating_points")
    if set(calibration) != {"operating_points"} or not isinstance(points, Mapping):
        raise TaggingError("must hold exactly one key, operating_points, an object")
    default = points.get("default")
    if set(points) != {"default"} or not isinstance(default, Mapping):
        raise TaggingError("operating_points must hold exactly one key, default, an object")
    biases = default.get("biases")
    if set(default) != {"biases"} or not isinstance(biases, Mapping):
        raise TaggingError("operating_points.default must hold exactly one key, biases, an object")
    if set(biases) != set(BIAS_KEYS):
        raise TaggingError(f"operating_points.default.biases must hold exactly {list(BIAS_KEYS)}")
    resolved: dict[str, float] = {}
    for key in BIAS_KEYS:
        value = biases[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TaggingError(f"operating_points.default.biases.{key} must be a number")
        if not math.isfinite(value):
            raise TaggingError(f"operating_points.default.biases.{key} must be finite")
        resolved[key] = float(value)
    return resolved


ZERO_BIASES: Mapping[str, float] = {key: 0.0 for key in BIAS_KEYS}


def _finite(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """The scores with NaN read as minus infinity (never chosen)."""
    return [[score if score == score else _NO_SCORE for score in row] for row in rows]


def _best(candidates: Iterable[tuple[float, int]]) -> tuple[float, int]:
    """The highest-scoring (score, index), the first on a tie; (-inf, -1)
    when there is none."""
    best = (_NO_SCORE, -1)
    for candidate in candidates:
        if candidate[0] > best[0]:
            best = candidate
    return best


def viterbi_path(
    rows: Sequence[Sequence[float]],
    tagset: TagSet,
    biases: Mapping[str, float] = ZERO_BIASES,
) -> list[int] | None:
    """The best label index per token under the scheme's constraints, given
    one row of label log-probabilities per token; None when no path is valid
    (a label set that cannot tag such a sequence at all).

    A path starts with ``O``, ``B`` or ``S`` and ends with ``O``, ``E`` or
    ``S``; after ``O``, ``E`` or ``S`` comes ``O``, ``B`` or ``S``; after
    ``B`` or ``I`` comes ``I`` or ``E`` of the same entity. Each allowed
    transition adds its bias. Linear in tokens × labels: every transition
    out of a closed state (``O``/``E``/``S``) depends only on whether it
    leaves the background, so its best predecessor is shared by all
    labels."""
    if not rows:
        return []
    tags, entities = tagset.tags, tagset.entities
    labels = range(len(tagset))
    emissions = _finite(rows)
    starts = ("O", "B", "S")
    score = [emissions[0][j] if tags[j] in starts else _NO_SCORE for j in labels]
    back: list[list[int]] = []
    stay = biases["transition_bias_background_stay"]
    enter = biases["transition_bias_background_to_start"]
    leave = biases["transition_bias_end_to_background"]
    again = biases["transition_bias_end_to_start"]
    keep_on = biases["transition_bias_inside_to_continue"]
    close = biases["transition_bias_inside_to_end"]
    for row in emissions[1:]:
        background = _best((score[j], j) for j in labels if tags[j] == "O")
        ended = _best((score[j], j) for j in labels if tags[j] in ("E", "S"))
        inside: dict[str, tuple[float, int]] = {}
        for j in labels:
            if tags[j] in ("B", "I") and score[j] > inside.get(entities[j], (_NO_SCORE, -1))[0]:
                inside[entities[j]] = (score[j], j)
        to_background = _best([(background[0] + stay, background[1]), (ended[0] + leave, ended[1])])
        to_start = _best([(background[0] + enter, background[1]), (ended[0] + again, ended[1])])
        new_score: list[float] = []
        pointers: list[int] = []
        for j in labels:
            tag = tags[j]
            if tag == "O":
                previous = to_background
            elif tag in ("B", "S"):
                previous = to_start
            elif tag in ("I", "E"):
                opened = inside.get(entities[j], (_NO_SCORE, -1))
                previous = (opened[0] + (keep_on if tag == "I" else close), opened[1])
            else:
                previous = (_NO_SCORE, -1)
            new_score.append(previous[0] + row[j])
            pointers.append(previous[1])
        score = new_score
        back.append(pointers)
    final, last = _best((score[j], j) for j in labels if tags[j] in ("O", "E", "S"))
    if final == _NO_SCORE:
        return None
    path = [last]
    for pointers in reversed(back):
        path.append(pointers[path[-1]])
    path.reverse()
    return path


def argmax_path(rows: Sequence[Sequence[float]]) -> list[int]:
    """Each token's best-scoring label index (the first on a tie)."""
    return [_best((score, j) for j, score in enumerate(row))[1] for row in _finite(rows)]


def spans_of(path: Sequence[int], tagset: TagSet) -> list[tuple[int, int, str]]:
    """The spans a label sequence marks, as ``(first, last, entity)`` token
    indices (``last`` inclusive), in order. ``B`` opens a span, ``I``
    continues one of its entity, ``E`` closes it, ``S`` is a span of one
    token; a tag that cannot continue the open span starts a new one; ``O``
    (or an untagged label) closes the open span, which is kept as it is."""
    spans: list[tuple[int, int, str]] = []
    first, entity = -1, ""

    def close(last: int) -> None:
        nonlocal first
        if first >= 0:
            spans.append((first, last, entity))
            first = -1

    for index, label in enumerate(path):
        tag = tagset.tags[label]
        if tag not in ("B", "I", "E", "S"):
            close(index - 1)
            continue
        if not (first >= 0 and tag in ("I", "E") and tagset.entities[label] == entity):
            close(index - 1)
            if tag == "S":
                spans.append((index, index, tagset.entities[label]))
                continue
            first, entity = index, tagset.entities[label]
        if tag == "E":
            close(index)
    close(len(path) - 1)
    return spans


Decoder = Callable[[Sequence[Sequence[float]]], list[int]]


def decoder_for(tagset: TagSet, biases: Mapping[str, float] | None) -> Decoder:
    """How a model's per-token scores become a label path: the constrained
    Viterbi decoder with ``biases`` (a model whose calibration the catalog
    lists), else the greedy reading of each token's best label. A sequence
    no valid path can tag falls back to the greedy reading."""
    if biases is None:
        return argmax_path

    def decode(rows: Sequence[Sequence[float]]) -> list[int]:
        path = viterbi_path(rows, tagset, biases)
        return path if path is not None else argmax_path(rows)

    return decode
