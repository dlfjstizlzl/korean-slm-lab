#!/usr/bin/env python3
"""Taxonomy, semantic plan shapes and Korean realization families.

Layering (see outputs/manager_dataset/02_task_taxonomy.md):

1. TAXONOMY           - what the request *means*: plan shape + argument kinds
2. plan_for(spec)     - canonical TaskPlan built only from the semantic spec
3. FAMILIES           - how a spec is *said*: one family = one syntactic frame
                        (argument placement, clause order, sentence type).
                        Inside a family only lexical choices vary.

Skill names appear here only as the targets of plan shapes. The generator
checks every one of them against the Runtime registry at start-up, so a skill
that disappears from slm-runtime makes generation fail instead of silently
producing plans the Runtime would reject.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------- #
# Korean particle helpers
# --------------------------------------------------------------------------- #


def _has_batchim(word: str) -> bool:
    last = word.rstrip()[-1]
    if "가" <= last <= "힣":
        return (ord(last) - 0xAC00) % 28 != 0
    # digits / latin fallbacks by pronunciation of the final character
    return last in "013678LMNlmn"


def _batchim_is_rieul(word: str) -> bool:
    last = word.rstrip()[-1]
    return "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 == 8


def eul(word: str) -> str:
    return "을" if _has_batchim(word) else "를"


def i_ga(word: str) -> str:
    return "이" if _has_batchim(word) else "가"


def eun(word: str) -> str:
    return "은" if _has_batchim(word) else "는"


def wa(word: str) -> str:
    return "과" if _has_batchim(word) else "와"


def iya(word: str) -> str:
    return "이야" if _has_batchim(word) else "야"


def ro(word: str) -> str:
    return "로" if not _has_batchim(word) or _batchim_is_rieul(word) else "으로"


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

REGISTERS = ("banmal", "polite", "formal", "question")

# (imperative stem, "-고" form, adnominal past form)
EXTRACT_VERBS = (
    ("뽑아", "뽑고", "뽑은"),
    ("추출해", "추출하고", "추출한"),
    ("골라", "고르고", "고른"),
    ("찾아", "찾고", "찾은"),
    ("짚어", "짚고", "짚은"),
    ("추려", "추리고", "추린"),
    ("목록으로 정리해", "목록으로 정리하고", "목록으로 정리한"),
    ("나열해", "나열하고", "나열한"),
)
EXTRACT_OBJECTS = ("핵심 키워드", "주요 키워드", "핵심 개념", "주요 개념", "중요한 용어",
                   "핵심어", "키워드", "주요 개념어", "핵심 용어", "중심 개념")
EXTRACT_NOUNS = ("추출", "목록 정리", "정리", "리스트 작성")

SUMMARY_VERBS = (
    ("요약해", "요약하고", "요약한"),
    ("한 문장으로 요약해", "한 문장으로 요약하고", "한 문장으로 요약한"),
    ("짧게 요약해", "짧게 요약하고", "짧게 요약한"),
    ("간단히 정리해", "간단히 정리하고", "간단히 정리한"),
    ("한 줄로 정리해", "한 줄로 정리하고", "한 줄로 정리한"),
    ("간추려", "간추리고", "간추린"),
    ("짧게 줄여", "짧게 줄이고", "짧게 줄인"),
    ("요점만 한 문장으로 정리해", "요점만 한 문장으로 정리하고", "요점만 한 문장으로 정리한"),
    ("핵심만 추려서 한 문장으로 정리해", "핵심만 추려서 한 문장으로 정리하고",
     "핵심만 추려서 한 문장으로 정리한"),
)
SUMMARY_NOUNS = ("요약", "한 줄 요약", "짧은 요약", "한 문장 요약")

THEN = (" 뒤", " 다음", " 후", " 다음에")

TEXT_DEIXIS = ("다음 글", "아래 글", "아래 내용", "다음 내용", "이 글", "아래 문단", "다음 텍스트",
               "아래 설명")
TEXT_NOUN = ("글", "내용", "문단", "설명", "텍스트", "단락")

PREFACES = (
    "발표 준비 중이야.", "보고서에 넣으려고 해요.", "스터디 자료로 쓸 거예요.", "시간이 별로 없어서요.",
    "메모해 두려고 해.", "내일 회의 전에 훑어보려고 합니다.", "블로그 초안을 쓰는 중인데,",
    "수업 복습용이야.", "팀원들한테 공유하려고요.", "퀴즈 문제를 만들려고 하는데,",
    "자료 조사하다가 발견했어.", "뉴스레터에 짧게 소개하려고 합니다.", "카드 뉴스로 만들 예정이라",
    "읽을 시간이 없어서 그래.",
)

ENDINGS = {
    "banmal": (" 줘.", "줘.", " 줘", "줘"),
    "polite": (" 주세요.", "주세요.", " 주세요", " 줄래요?"),
    "formal": (" 주시겠어요?", " 주시기 바랍니다.", " 주시면 감사하겠습니다.", " 주십시오."),
    "question": (" 줄래?", " 줄 수 있어?", "줄래?", " 줄 수 있을까?"),
}
REQUEST_WORDS = {
    "banmal": ("부탁해.", "부탁해", "해 줘."),
    "polite": ("부탁해요.", "부탁드려요.", "해 주세요."),
    "formal": ("부탁드립니다.", "요청드립니다.", "부탁드리겠습니다."),
    "question": ("해 줄래?", "가능해?", "부탁해도 될까?"),
}
COPULA = {"banmal": iya, "polite": lambda w: "이에요" if _has_batchim(w) else "예요",
          "formal": lambda w: "입니다", "question": iya}
WHAT = {"banmal": "뭐야?", "polite": "뭔가요?", "formal": "무엇인가요?", "question": "뭐야?"}
NEED = {"banmal": "필요해.", "polite": "필요해요.", "formal": "필요합니다.", "question": "필요한데 해 줄래?"}


@dataclass
class Realizer:
    """Lexical choices for one request. Register is fixed per request."""

    rng: random.Random
    register: str
    args: dict[str, str]
    trace: list[str] = field(default_factory=list)

    def pick(self, options, tag: str):
        index = self.rng.randrange(len(options))
        self.trace.append(f"{tag}{index}")
        return options[index]

    # arguments ---------------------------------------------------------
    @property
    def T(self) -> str:
        return self.args["topic"]

    @property
    def X(self) -> str:
        return self.args["text"]

    # lexical helpers ---------------------------------------------------
    def end(self, stem: str) -> str:
        return stem + self.pick(ENDINGS[self.register], "e")

    def req(self) -> str:
        return self.pick(REQUEST_WORDS[self.register], "r")

    def obj(self) -> str:
        return self.pick(EXTRACT_OBJECTS, "o")

    def ext_verb(self):
        return self.pick(EXTRACT_VERBS, "xv")

    def sum_verb(self):
        return self.pick(SUMMARY_VERBS, "sv")

    def ext(self, obj: str | None = None, only: bool = False) -> str:
        obj = obj or self.obj()
        head = obj + ("만" if only else eul(obj))
        return f"{head} {self.end(self.ext_verb()[0])}"

    def ext_go(self, obj: str | None = None) -> str:
        obj = obj or self.obj()
        return f"{obj}{eul(obj)} {self.ext_verb()[1]}"

    def ext_then(self, obj: str | None = None) -> str:
        obj = obj or self.obj()
        return f"{obj}{eul(obj)} {self.ext_verb()[2]}{self.pick(THEN, 't')}"

    def summ(self) -> str:
        return self.end(self.sum_verb()[0])

    def sum_go(self) -> str:
        return self.sum_verb()[1]

    def sum_then(self) -> str:
        return self.sum_verb()[2] + self.pick(THEN, "t")

    def deixis(self) -> str:
        return self.pick(TEXT_DEIXIS, "d")

    def noun(self) -> str:
        return self.pick(TEXT_NOUN, "n")

    def pre(self) -> str:
        return self.pick(PREFACES, "p")

    def ext_noun(self) -> str:
        return self.pick(EXTRACT_NOUNS, "xn")

    def sum_noun(self) -> str:
        return self.pick(SUMMARY_NOUNS, "sn")

    def cop(self, word: str) -> str:
        return word + COPULA[self.register](word)


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Taxonomy:
    name: str
    group: str  # normal | hard_labeled | policy_pending
    description: str
    steps: tuple[tuple[str, str], ...]  # (skill, source) source = arg name | "task:N"
    args: tuple[str, ...]
    quota: int
    difficulty: str
    hard_case_category: str | None = None
    candidate_plans: tuple[tuple[str, tuple[tuple[str, str], ...] | None], ...] = ()
    notes: str = ""


TAXONOMY: tuple[Taxonomy, ...] = (
    Taxonomy("single_search_topic", "normal",
             "Extract key concepts/keywords for a named topic; no text supplied.",
             (("search", "topic"),), ("topic",), 22, "easy",
             notes="Mirrors runtime few-shot 1. The search skill only sees the topic string."),
    Taxonomy("single_search_text", "normal",
             "Extract key concepts/keywords from text supplied in the request.",
             (("search", "text"),), ("text",), 26, "easy"),
    Taxonomy("single_summary_text", "normal",
             "Summarize text supplied in the request.",
             (("summary", "text"),), ("text",), 34, "easy",
             notes="Mirrors runtime few-shot 2."),
    Taxonomy("single_summary_topic", "normal",
             "Summarize a named topic with no supplied text.",
             (("summary", "topic"),), ("topic",), 16, "easy",
             notes="Mirrors runtime few-shot 4; the summary skill must supply content itself "
                   "(REQUIRES HUMAN DECISION whether this is desirable)."),
    Taxonomy("chain_search_then_summary", "normal",
             "Extract key concepts (from a topic or supplied text), then summarize the extracted "
             "concepts. Task 1 consumes task 0's output.",
             (("search", "source"), ("summary", "task:0")), ("source",), 38, "medium",
             notes="Mirrors runtime few-shot 3. `source` is a topic or a supplied text, fixed by "
                   "the realization family."),
    Taxonomy("chain_summary_then_search", "normal",
             "Summarize (a supplied text or topic), then extract keywords from that summary.",
             (("summary", "source"), ("search", "task:0")), ("source",), 28, "medium"),
    Taxonomy("trap_keyword_misroute", "hard_labeled",
             "Surface words of one skill (요약/검색) appear, but the requested operation is the "
             "other skill.",
             (("?", "text"),), ("text",), 8, "hard", hard_case_category="keyword_misroute",
             notes="The target skill is fixed per family."),
    Taxonomy("trap_unnecessary_skill", "hard_labeled",
             "Request mentions a second operation only to exclude or qualify it; exactly one task "
             "is correct.",
             (("?", "text"),), ("text",), 7, "hard", hard_case_category="unnecessary_skill"),
    Taxonomy("trap_order_inversion", "hard_labeled",
             "Clause order on the surface is the reverse of execution order; a two-task chain is "
             "correct.",
             (("?", "text"), ("?", "task:0")), ("text",), 7, "hard",
             hard_case_category="order_inversion"),
    Taxonomy("policy_unsupported_capability", "policy_pending",
             "Needs a capability no registered skill provides (SQL, code, translation, live web "
             "retrieval, factual QA).",
             (), (), 5, "hard", hard_case_category="unsupported_capability",
             candidate_plans=(("reject_before_planning", None),
                              ("nearest_skill_fallback", (("search", "request"),)))),
    Taxonomy("policy_ambiguous", "policy_pending",
             "Missing payload or unclear operation; no single correct plan under current policy.",
             (), (), 4, "hard", hard_case_category="ambiguous",
             candidate_plans=(("clarification_outside_plan", None),
                              ("best_guess_plan", (("summary", "request"),)))),
    Taxonomy("policy_cross_request_reference", "policy_pending",
             "Refers to a result from an earlier request; ManagerClient::plan is stateless.",
             (), (), 2, "hard", hard_case_category="cross_request_reference",
             candidate_plans=(("reject_no_session_state", None),)),
    Taxonomy("policy_multi_independent", "policy_pending",
             "Two independent operations on two texts. Runtime-valid plan exists, but only the "
             "last task's output reaches the user.",
             (), ("text", "text2"), 3, "hard", hard_case_category="multi_independent",
             candidate_plans=(("independent_tasks_last_output_only",
                               (("summary", "text"), ("search", "text2"))),
                              ("reject_until_multi_output_supported", None))),
)

TAXONOMY_BY_NAME = {item.name: item for item in TAXONOMY}


def referenced_skills() -> set[str]:
    skills = set()
    for item in TAXONOMY:
        skills.update(skill for skill, _ in item.steps if skill != "?")
        for _, plan in item.candidate_plans:
            if plan:
                skills.update(skill for skill, _ in plan)
    for family in FAMILIES:
        if family.steps:
            skills.update(skill for skill, _ in family.steps)
    return skills


# --------------------------------------------------------------------------- #
# Realization families
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Family:
    id: str
    taxonomy: str
    source: str  # topic | text | none | text2 | authored
    render: Callable[[Realizer], str]
    # trap families fix their own plan shape because the taxonomy only says "?"
    steps: tuple[tuple[str, str], ...] | None = None
    authored: tuple[str, ...] = ()
    note: str = ""


def sent(text: str) -> str:
    """Close a clause that is followed by more text."""
    return text if text.endswith((".", "?", "!")) else text + "."


def tidy(text: str) -> str:
    """Punctuation clean-up applied to every rendered request."""
    import re

    text = re.sub(r"\.:(\s)", r":\1", text)
    text = re.sub(r"\?:(\s)", r"?\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _join(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _topic_rel(r: Realizer) -> str:
    return r.pick((f"{r.T}의", f"{r.T}에서", f"{r.T}에 관한", f"{r.T}{wa(r.T)} 관련된"), "rel")


F = Family
FAMILIES: tuple[Family, ...] = (
    # ---------------------------------------------------------- single_search_topic
    F("st_direct", "single_search_topic", "topic", lambda r: f"{_topic_rel(r)} {r.ext()}"),
    F("st_study_question", "single_search_topic", "topic",
      lambda r: (lambda o: f"{r.T}{eul(r.T)} 공부하려는데 어떤 {o}{i_ga(o)} 중요한지 {r.end('알려')}")(r.obj())),
    F("st_preface", "single_search_topic", "topic",
      lambda r: _join(r.pre(), f"{r.T} 관련 {r.ext(only=True)}")),
    F("st_topic_last", "single_search_topic", "topic",
      lambda r: f"{sent(r.ext())} 주제는 {r.cop(r.T)}." if r.register != "question"
      else f"주제는 {r.cop(r.T)}. {r.ext()}"),
    F("st_noun_request", "single_search_topic", "topic",
      lambda r: f"{r.T} {r.obj()} {r.ext_noun()} {r.req()}"),
    F("st_dash_label", "single_search_topic", "topic",
      lambda r: f"{r.T} — 이 주제의 {r.ext()}"),
    F("st_before_writing", "single_search_topic", "topic",
      lambda r: f"{r.T}에 대해 글을 쓰기 전에 꼭 들어가야 할 {r.ext()}"),
    # ---------------------------------------------------------- single_search_text
    F("sx_colon", "single_search_text", "text", lambda r: f"{r.deixis()}에서 {r.ext()}: {r.X}"),
    F("sx_quote_first", "single_search_text", "text",
      lambda r: f"\"{r.X}\" — 이 {r.noun()}에 나오는 {r.ext(only=True)}"),
    F("sx_text_first_newline", "single_search_text", "text",
      lambda r: f"{r.X}\n\n위 {r.noun()}의 {r.ext()}"),
    F("sx_preface", "single_search_text", "text",
      lambda r: _join(r.pre(), f"아래 {r.noun()}에서 {sent(r.ext())} {r.X}")),
    F("sx_what_question", "single_search_text", "text",
      lambda r: (lambda o: f"이 {r.noun()}의 {o}{i_ga(o)} {WHAT[r.register]} {r.X}")(r.obj())),
    F("sx_read_then_list", "single_search_text", "text",
      lambda r: (lambda n, o: f"아래 {n}{eul(n)} 읽고 {o}{eul(o)} 목록으로 {r.end('정리해')}: {r.X}")(r.noun(), r.obj())),
    F("sx_bracket_noun", "single_search_text", "text",
      lambda r: f"[{r.X}] 여기서 {r.obj()} {r.ext_noun()} {r.req()}"),
    # ---------------------------------------------------------- single_summary_text
    F("mx_colon", "single_summary_text", "text",
      lambda r: (lambda d: f"{d}{eul(d)} {r.summ()}: {r.X}")(r.deixis())),
    F("mx_quote_first", "single_summary_text", "text",
      lambda r: f"\"{r.X}\" 이 {r.noun()} 좀 {r.summ()}"),
    F("mx_text_first_newline", "single_summary_text", "text",
      lambda r: (lambda n: f"{r.X}\n\n위 {n}{eul(n)} {r.summ()}")(r.noun())),
    F("mx_preface", "single_summary_text", "text",
      lambda r: (lambda n: _join(r.pre(), f"이 {n}{eul(n)} {sent(r.summ())} {r.X}"))(r.noun())),
    F("mx_gist", "single_summary_text", "text",
      lambda r: f"다음 {r.noun()}의 요지가 뭔지 한 문장으로 {r.end('알려')}: {r.X}"),
    F("mx_noun_request", "single_summary_text", "text",
      lambda r: f"{r.X} — {r.sum_noun()} {r.req()}"),
    F("mx_read_and_tell", "single_summary_text", "text",
      lambda r: (lambda n: f"아래 {n}{eul(n)} 읽고 무슨 내용인지 짧게 {r.end('말해')}: {r.X}")(r.noun())),
    # ---------------------------------------------------------- single_summary_topic
    F("mt_direct", "single_summary_topic", "topic", lambda r: f"{r.T}{eul(r.T)} {r.summ()}"),
    F("mt_preface", "single_summary_topic", "topic",
      lambda r: _join(r.pre(), f"{r.T}{eul(r.T)} {r.summ()}")),
    F("mt_noun_request", "single_summary_topic", "topic",
      lambda r: f"{r.T} {r.sum_noun()} {r.req()}"),
    F("mt_topic_last", "single_summary_topic", "topic",
      lambda r: f"{sent(r.summ())} 대상은 {r.cop(r.T)}." if r.register != "question"
      else f"대상은 {r.cop(r.T)}. {r.summ()}"),
    F("mt_need", "single_summary_topic", "topic",
      lambda r: f"{r.T}에 대한 짧은 요약이 {NEED[r.register]}"),
    F("mt_core_one_line", "single_summary_topic", "topic",
      lambda r: f"{r.T}, 핵심만 한 줄로 {r.end('정리해')}"),
    F("mt_two_sentences", "single_summary_topic", "topic",
      lambda r: f"{r.T}{eul(r.T)} 한두 문장으로 요약하면 어떻게 되는지 {r.end('보여')}"),
    # ---------------------------------------------------------- chain_search_then_summary
    F("cs_go_topic", "chain_search_then_summary", "topic",
      lambda r: f"{r.T}의 {r.ext_go()} 그걸 {r.summ()}"),
    F("cs_then_topic", "chain_search_then_summary", "topic",
      lambda r: f"{r.T}에서 {r.ext_then()}, 그 결과를 바탕으로 {r.summ()}"),
    F("cs_two_steps_topic", "chain_search_then_summary", "topic",
      lambda r: (lambda o: f"두 단계로 {sent(r.req())} 먼저 {r.T}의 {o} 추출, 다음으로 추출한 {o}{eul(o)} "
                           f"한 문장으로 요약.")(r.obj())),
    F("cs_colon_text", "chain_search_then_summary", "text",
      lambda r: f"{r.deixis()}에서 {r.ext_go()}, 뽑은 내용만 가지고 {r.summ()}: {r.X}"),
    F("cs_text_first_newline", "chain_search_then_summary", "text",
      lambda r: (lambda n, o: f"{r.X}\n\n위 {n}의 {o}{eul(o)} 먼저 {r.ext_verb()[1]}, "
                              f"그 {o}{eul(o)} 바탕으로 {r.summ()}")(r.noun(), r.obj())),
    F("cs_preface_text", "chain_search_then_summary", "text",
      lambda r: _join(r.pre(), f"이 {r.noun()}에서 {r.ext_then()} 그것들로 짧은 요약을 "
                               f"{sent(r.end('만들어'))} {r.X}")),
    F("cs_concepts_to_sentence_topic", "chain_search_then_summary", "topic",
      lambda r: (lambda o: f"{r.T}의 {o}{eul(o)} {r.ext_verb()[1]}, 그 {o}{eul(o)} 이어서 한 문장 요약으로 "
                           f"{r.end('만들어')}")(r.obj())),
    # ---------------------------------------------------------- chain_summary_then_search
    F("sc_go_text", "chain_summary_then_search", "text",
      lambda r: (lambda n: f"이 {n}{eul(n)} {r.sum_go()} 그 요약문에서 {r.ext()}: {r.X}")(r.noun())),
    F("sc_text_first_newline", "chain_summary_then_search", "text",
      lambda r: f"{r.X}\n\n위 내용을 {r.sum_then()} 요약한 문장에서 {r.ext(only=True)}"),
    F("sc_numbered_steps", "chain_summary_then_search", "text",
      lambda r: (lambda n: f"순서대로 {sent(r.req())} ① 아래 {n} 요약 ② 요약본에서 {r.obj()} 추출. {r.X}")(r.noun())),
    F("sc_preface_text", "chain_summary_then_search", "text",
      lambda r: (lambda n: _join(r.pre(), f"아래 {n}{eul(n)} {r.sum_then()} 줄인 내용에 들어 있는 "
                                          f"{sent(r.ext())} {r.X}"))(r.noun())),
    F("sc_topic", "chain_summary_then_search", "topic",
      lambda r: f"{r.T}{eul(r.T)} {r.sum_then()} 그 요약에서 {r.ext()}"),
    F("sc_what_remains", "chain_summary_then_search", "text",
      lambda r: (lambda n, o: f"다음 {n}{eul(n)} 한 문장으로 줄이면 어떤 {o}{i_ga(o)} 남을까? "
                              f"요약한 뒤 거기서 {r.end(r.ext_verb()[0])}: {r.X}")(r.noun(), r.obj())),
    F("sc_after_summary_colon", "chain_summary_then_search", "text",
      lambda r: (lambda d: f"{d}{eul(d)} {r.sum_go()} 나서 요약 결과의 {r.ext()}: {r.X}")(r.deixis())),
    # ---------------------------------------------------------- traps
    F("kt_summary_doc_extract", "trap_keyword_misroute", "text",
      lambda r: f"아래 요약문에서 {r.ext(only=True)}: {r.X}", steps=(("search", "text"),),
      note="contains 요약, wants extraction"),
    F("kt_already_summarized", "trap_keyword_misroute", "text",
      lambda r: f"요약은 이미 돼 있어. 이 {r.noun()}에서 {sent(r.ext(only=True))} {r.X}",
      steps=(("search", "text"),), note="contains 요약, wants extraction"),
    F("kt_search_result_shorten", "trap_keyword_misroute", "text",
      lambda r: f"검색해서 찾은 설명인데 너무 길어. {sent(r.summ())} {r.X}", steps=(("summary", "text"),),
      note="contains 검색/찾은, wants summary"),
    F("kt_keyword_search_output", "trap_keyword_misroute", "text",
      lambda r: (lambda n: f"키워드 검색 결과로 나온 {r.cop(n)}. 한 문장으로 "
                           f"{r.end('간추려')}: {r.X}")(r.noun()), steps=(("summary", "text"),),
      note="contains 키워드/검색, wants summary"),
    F("ut_no_keywords", "trap_unnecessary_skill", "text",
      lambda r: f"키워드는 필요 없어. {r.summ()}: {r.X}", steps=(("summary", "text"),)),
    F("ut_do_not_summarize", "trap_unnecessary_skill", "text",
      lambda r: f"요약하지 말고 {r.ext(only=True)}: {r.X}", steps=(("search", "text"),)),
    F("ut_summary_with_concepts", "trap_unnecessary_skill", "text",
      lambda r: f"핵심 개념이 잘 드러나게 {r.summ()}: {r.X}", steps=(("summary", "text"),),
      note="mentions 핵심 개념 as a quality of the summary, not a separate task"),
    F("ot_summary_first_but_extract_before", "trap_order_inversion", "text",
      lambda r: (lambda o: f"{sent(r.summ())} 단, 요약하기 전에 {o}부터 {r.ext_verb()[1]} 그 {o}만으로 요약해야 해: "
                           f"{r.X}")(r.obj()),
      steps=(("search", "text"), ("summary", "task:0"))),
    F("ot_extract_from_summary_not_source", "trap_order_inversion", "text",
      lambda r: (lambda n: f"{sent(r.ext())} 원문 말고, 이 {n}{eul(n)} 한 문장으로 요약한 결과에서 뽑아야 해: "
                           f"{r.X}")(r.noun()),
      steps=(("summary", "text"), ("search", "task:0"))),
    F("ot_goal_then_prerequisite", "trap_order_inversion", "text",
      lambda r: (lambda o: f"요약본 기준으로 {o}{eul(o)} 뽑고 싶어. 그러려면 우선 이 {r.noun()}부터 "
                           f"{sent(r.summ())} {r.X}")(r.obj()),
      steps=(("summary", "text"), ("search", "task:0"))),
    # ---------------------------------------------------------- policy pending
    F("un_sql", "policy_unsupported_capability", "authored", lambda r: r.args["authored"],
      authored=("지난달 가입한 사용자 수를 구하는 SQL 쿼리 좀 짜 줘.",
                "orders 테이블에서 금액이 가장 큰 주문 열 건을 조회해 주세요.")),
    F("un_code", "policy_unsupported_capability", "authored", lambda r: r.args["authored"],
      authored=("파이썬으로 리스트를 역순 정렬하는 함수 작성해 줄래?",)),
    F("un_translate", "policy_unsupported_capability", "text",
      lambda r: (lambda n: f"다음 {n}{eul(n)} 영어로 {r.end('번역해')}: {r.X}")(r.noun())),
    F("un_live_web", "policy_unsupported_capability", "topic",
      lambda r: f"{r.T} 관련 오늘자 최신 뉴스를 인터넷에서 검색해 줘.",
      note="contains 검색 but the search skill cannot retrieve"),
    F("un_factual_qa", "policy_unsupported_capability", "topic",
      lambda r: f"{r.T}{i_ga(r.T)} 언제 처음 알려졌는지 정확한 연도를 알려 줘."),
    F("am_missing_payload", "policy_ambiguous", "authored", lambda r: r.args["authored"],
      authored=("이거 요약해 줘.", "아래 글에서 키워드 좀 뽑아 줄래?", "방금 붙여 넣은 내용 정리 부탁해요.")),
    F("am_unclear_operation", "policy_ambiguous", "text", lambda r: f"{r.X} 이거 좀 봐 줘."),
    F("cr_previous_result", "policy_cross_request_reference", "authored", lambda r: r.args["authored"],
      authored=("방금 뽑은 키워드로 한 문장 요약 만들어 줘.", "아까 요약한 거에서 핵심 개념만 다시 뽑아 줘.",
                "이전 결과를 조금 더 짧게 줄여 줄 수 있어?")),
    F("mi_two_texts", "policy_multi_independent", "text2",
      lambda r: (lambda o: f"두 가지 {sent(r.req())} 첫 번째 글은 {r.sum_noun()}, 두 번째 글은 {o} 추출. "
                           f"1) {r.X} 2) {r.args['text2']}")(r.obj())),
)

FAMILY_BY_ID = {family.id: family for family in FAMILIES}


def families_for(taxonomy: str) -> list[Family]:
    return [family for family in FAMILIES if family.taxonomy == taxonomy]
