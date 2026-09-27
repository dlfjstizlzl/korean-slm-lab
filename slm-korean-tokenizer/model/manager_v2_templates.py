#!/usr/bin/env python3
"""Taxonomy, semantic intents and Korean realization families (TaskPlan v2).

Layers
------
1. INTENTS[name](slots) -> canonical TaskPlan. Uses only semantic slots.
2. TAXONOMY            -> what a request type means, pilot quota, group.
3. FAMILIES            -> how a spec is said. One family = one syntactic frame
                          (clause order, argument placement, sentence type).
                          Only lexical slots vary inside a family.

Anti-shortcut design (targets the two Pilot v1 failures):
* Skill cue words are shared across skills: 뽑아 is used for extract *and* sql,
  찾아 for search, extract *and* sql, 정리 for summary *and* extract, 확인 for
  search *and* sql, 알려 for everything. Retrieval is often requested without
  any search verb; hard families put 검색/요약 in requests that need neither.
* Verbosity (context prefaces/suffixes) is drawn independently of the plan, and
  dedicated families produce short multi-task and long single-task requests.

Skill names appear only inside INTENTS. The generator checks them against the
Runtime registry (`slm_taskplan_tool --contract`) before generating anything.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

from manager_dataset_templates import eul, eun, i_ga, wa  # v1 particle helpers (batchim rules)
from manager_v2_content import CLARIFY_MESSAGES, UNSUPPORTED_MESSAGE

# --------------------------------------------------------------------------- #
# Canonical plans
# --------------------------------------------------------------------------- #


def _t(skill: str, instruction: str, sources: list) -> dict:
    return {"skill": skill, "instruction": instruction, "sources": sources}


def _plan(tasks: list[dict], outputs: list[int]) -> dict:
    return {"status": "ok",
            "tasks": [{"id": i, **task} for i, task in enumerate(tasks)],
            "outputs": outputs}


def _d(s, text: str, prefix: bool = True) -> str:
    """Attach an optional single-task constraint slot D (revision r2)."""
    if not s.get("D"):
        return text
    return f"{s['D']} {text}" if prefix else f"{text}, {s['D']}"


def _search(s, t="T", a="A"):
    return _t("search", _d(s, f"{s[t]} {s[a]} 검색") if t == "T" else f"{s[t]} {s[a]} 검색", [])


def _sql(s, q="Q"):
    return _t("sql", _d(s, f"{s[q]} 조회", prefix=False) if q == "Q" else f"{s[q]} 조회", [])


def _search_sql_pair(s):
    search = _t("search", f"{s['E']} {s['AE']} 검색", [])
    sql = _t("sql", f"{s['QI']} 조회", [])
    return [sql, search] if s.get("order") == "sql_first" else [search, sql]


INTENTS: dict[str, Callable[[dict], dict]] = {
    "search": lambda s: _plan([_search(s)], [0]),
    "search_summary_doc": lambda s: _plan([_t("search", f"{s['T']} {s['A']} 요약 자료 검색", [])], [0]),
    "summary_user": lambda s: _plan([_t("summary", _d(s, "핵심 내용 요약"), ["user"])], [0]),
    "extract_user": lambda s: _plan([_t("extract", _d(s, f"{s['F']} 추출", prefix=False), ["user"])], [0]),
    "sql": lambda s: _plan([_sql(s)], [0]),
    "compare_user": lambda s: _plan([_t("compare", f"{s['TT1']}{wa(s['TT1'])} {s['TT2']}의 {s['C']} 비교",
                                        ["user"])], [0]),
    "search_summary": lambda s: _plan([_search(s), _t("summary", f"{s['A']} 요약", [0])], [1]),
    "search_extract": lambda s: _plan([_search(s), _t("extract", f"{s['F']} 추출", [0])], [1]),
    "sql_summary": lambda s: _plan([_sql(s), _t("summary", "조회 결과 요약", [0])], [1]),
    "sql_extract": lambda s: _plan([_sql(s), _t("extract", f"{s['F']} 추출", [0])], [1]),
    "extract_summary_user": lambda s: _plan([_t("extract", f"{s['F']} 추출", ["user"]),
                                             _t("summary", f"추출한 {s['F']} 요약", [0])], [1]),
    "search_search_compare": lambda s: _plan([_search(s, "T1"), _search(s, "T2"),
                                              _t("compare", f"{s['C']} 비교", [0, 1])], [2]),
    "sql_sql_compare": lambda s: _plan([_sql(s, "Q1"), _sql(s, "Q2"),
                                        _t("compare", f"{s['C']} 비교", [0, 1])], [2]),
    "search_sql_compare": lambda s: _plan(_search_sql_pair(s) + [_t("compare", f"{s['CI']} 비교", [0, 1])], [2]),
    "user_search_compare": lambda s: _plan([_t("search", f"{s['T']} 최신 정보 검색", []),
                                            _t("compare", "글 내용과 최신 정보 비교", ["user", 0])], [1]),
    "ssc_summary": lambda s: _plan([_search(s, "T1"), _search(s, "T2"),
                                    _t("compare", f"{s['C']} 비교", [0, 1]),
                                    _t("summary", "비교 결과 요약", [2])], [3]),
    "search_extract_summary": lambda s: _plan([_search(s), _t("extract", f"{s['F']} 추출", [0]),
                                               _t("summary", f"추출한 {s['F']} 요약", [1])], [2]),
    "sql_extract_summary": lambda s: _plan([_sql(s), _t("extract", f"{s['F']} 추출", [0]),
                                            _t("summary", f"추출한 {s['F']} 요약", [1])], [2]),
    "mo_search_summary_extract": lambda s: _plan([_search(s), _t("summary", f"{s['A']} 요약", [0]),
                                                  _t("extract", f"{s['F']} 추출", [0])], [1, 2]),
    "mo_user_summary_extract": lambda s: _plan([_t("summary", "핵심 내용 요약", ["user"]),
                                                _t("extract", f"{s['F']} 추출", ["user"])], [0, 1]),
    "mo_search_sql": lambda s: _plan(_search_sql_pair(s), [0, 1]),
    "mo_search_search": lambda s: _plan([_search(s, "T1"), _search(s, "T2", "A2")], [0, 1]),
    "clarify": lambda s: {"status": "clarify", "message": CLARIFY_MESSAGES[s["reason"]], "tasks": [], "outputs": []},
    "unsupported": lambda s: {"status": "unsupported", "message": UNSUPPORTED_MESSAGE, "tasks": [], "outputs": []},
}

# slots whose values a canonical instruction uses and that the request must state
# (checked by the generator, spaces ignored); derived composites are exempt
GROUNDED_SLOTS = {
    "search": ("T", "A", "D"), "search_summary_doc": ("T", "A"), "search_summary": ("T", "A"),
    "search_extract": ("T", "A", "F"), "extract_user": ("F", "D"), "sql": ("Q", "D"), "compare_user": ("TT1", "TT2", "C"),
    "sql_summary": ("Q",), "sql_extract": ("Q", "F"), "extract_summary_user": ("F",),
    "search_search_compare": ("T1", "T2", "C"), "sql_sql_compare": ("Q1", "Q2"),
    "search_sql_compare": ("E", "AE", "QI"), "user_search_compare": ("T",), "ssc_summary": ("T1", "T2", "C"),
    "search_extract_summary": ("T", "A", "F"), "sql_extract_summary": ("Q", "F"),
    "mo_search_summary_extract": ("T", "A", "F"), "mo_user_summary_extract": ("F",),
    "mo_search_sql": ("E", "AE", "QI"), "mo_search_search": ("T1", "A", "T2", "A2"),
    "summary_user": ("D",), "clarify": (), "unsupported": (),
}

# which slots each intent needs (the generator fills them)
INTENT_SLOTS = {
    "search": "search", "search_summary_doc": "search", "search_summary": "search",
    "search_extract": "search_fields", "search_extract_summary": "search_fields",
    "mo_search_summary_extract": "search_fields",
    "summary_user": "payload", "extract_user": "payload_fields", "extract_summary_user": "payload_fields",
    "mo_user_summary_extract": "payload_fields", "compare_user": "payload_pair",
    "sql": "sql", "sql_summary": "sql", "sql_extract": "sql_fields", "sql_extract_summary": "sql_fields",
    "search_search_compare": "search_pair", "ssc_summary": "search_pair", "mo_search_search": "search_pair2",
    "sql_sql_compare": "sql_pair", "search_sql_compare": "internal", "mo_search_sql": "internal",
    "user_search_compare": "payload_topic", "clarify": "clarify", "unsupported": "unsupported",
}

# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

REGISTERS = ("banmal", "polite", "formal", "question")
ENDINGS = {
    "banmal": (" 줘", "줘", " 줘.", "줘."),
    "polite": (" 주세요", " 주세요.", "주세요.", " 줄래요?"),
    "formal": (" 주시겠어요?", " 주시기 바랍니다.", " 주십시오.", " 주시면 감사하겠습니다."),
    "question": (" 줄래?", " 줄 수 있어?", "줄래?", " 줄 수 있을까?"),
}
REQUEST = {
    "banmal": ("부탁해.", "해 줘.", "부탁할게."),
    "polite": ("부탁해요.", "부탁드려요.", "해 주세요."),
    "formal": ("부탁드립니다.", "요청드립니다.", "부탁드리겠습니다."),
    "question": ("해 줄래?", "가능해?", "부탁해도 될까?"),
}
# (imperative 아/어 stem, -고, -아서, past adnominal, action noun)
SEARCH_V = (("찾아", "찾고", "찾아서", "찾은", "검색"), ("검색해", "검색하고", "검색해서", "검색한", "검색"),
            ("조사해", "조사하고", "조사해서", "조사한", "조사"), ("알아봐", "알아보고", "알아봐서", "알아본", "조사"),
            ("찾아봐", "찾아보고", "찾아봐서", "찾아본", "검색"), ("확인해", "확인하고", "확인해서", "확인한", "확인"),
            ("검색해 봐", "검색해 보고", "검색해 봐서", "검색해 본", "검색"))
SQL_V = (("조회해", "조회하고", "조회해서", "조회한", "조회"), ("뽑아", "뽑고", "뽑아서", "뽑은", "조회"),
         ("가져와", "가져오고", "가져와서", "가져온", "조회"), ("집계해", "집계하고", "집계해서", "집계한", "집계"),
         ("확인해", "확인하고", "확인해서", "확인한", "확인"), ("찾아", "찾고", "찾아서", "찾은", "조회"),
         ("뽑아 봐", "뽑아 보고", "뽑아 봐서", "뽑아 본", "조회"))
SUMMARY_V = (("요약해", "요약하고", "요약해서", "요약한", "요약"), ("정리해", "정리하고", "정리해서", "정리한", "정리"),
             ("간추려", "간추리고", "간추려서", "간추린", "요약"),
             ("짧게 줄여", "짧게 줄이고", "짧게 줄여서", "짧게 줄인", "요약"),
             ("핵심만 정리해", "핵심만 정리하고", "핵심만 정리해서", "핵심만 정리한", "요약"),
             ("한 문단으로 정리해", "한 문단으로 정리하고", "한 문단으로 정리해서", "한 문단으로 정리한", "요약"),
             ("요점만 추려", "요점만 추리고", "요점만 추려서", "요점만 추린", "요약"),
             ("간단히 요약해", "간단히 요약하고", "간단히 요약해서", "간단히 요약한", "요약"))
EXTRACT_V = (("뽑아", "뽑고", "뽑아서", "뽑은", "추출"), ("추출해", "추출하고", "추출해서", "추출한", "추출"),
             ("골라", "고르고", "골라서", "고른", "추출"), ("찾아", "찾고", "찾아서", "찾은", "추출"),
             ("정리해", "정리하고", "정리해서", "정리한", "정리"), ("적어", "적고", "적어서", "적은", "정리"),
             ("추려", "추리고", "추려서", "추린", "추출"))
COMPARE_V = (("비교해", "비교하고", "비교해서", "비교한", "비교"), ("견줘 봐", "견줘 보고", "견줘 봐서", "견줘 본", "비교"),
             ("대조해", "대조하고", "대조해서", "대조한", "비교"), ("나란히 비교해", "나란히 비교하고", "나란히 비교해서",
                                                             "나란히 비교한", "비교"),
             ("따져 봐", "따져 보고", "따져 봐서", "따져 본", "비교"))
THEN = (" 뒤", " 다음", " 후", " 다음에")
USER_REF = ("이 글", "아래 내용", "다음 글", "붙여 넣은 문서", "이 자료", "아래 기사", "이 문단", "아래 텍스트")
PREV_REF = {"search": ("찾은 내용", "검색 결과", "그 자료", "찾은 자료"),
            "sql": ("조회 결과", "그 데이터", "뽑은 결과", "조회한 수치"),
            "extract": ("뽑은 항목", "추출한 내용", "그 항목들"),
            "compare": ("비교 결과", "비교한 내용")}
GENERIC_REF = ("그 결과", "그것", "그 내용")


@dataclass
class R:
    """Lexical choices for one request; register is fixed per request."""

    rng: random.Random
    register: str
    trace: list = field(default_factory=list)

    def pick(self, options, tag="x"):
        index = self.rng.randrange(len(options))
        self.trace.append(f"{tag}{index}")
        return options[index]

    def end(self, stem: str) -> str:
        return stem + self.pick(ENDINGS[self.register], "e")

    def req(self) -> str:
        return self.pick(REQUEST[self.register], "r")

    def ref(self, op: str) -> str:
        return self.pick(PREV_REF.get(op, ()) + GENERIC_REF, "ref")

    def uref(self) -> str:
        return self.pick(USER_REF, "u")


def sent(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else text + "."


def tidy(text: str) -> str:
    text = re.sub(r"\.:(\s)", r":\1", text)
    text = re.sub(r"\?:(\s)", r"?\1", text)
    text = re.sub(r"\.\.", ".", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _form(r: R, table, form: str) -> str:
    stem, go, seo, adn, _ = r.pick(table, "v")
    return {"final": r.end(stem), "go": go, "seo": seo, "then": adn + r.pick(THEN, "t"), "plain": stem}[form]


def _noun(r: R, table) -> str:
    return r.pick(table, "v")[4]


# ---- per-skill phrases ------------------------------------------------------

def s_obj(r: R, t: str, a: str) -> str:
    return r.pick((f"{t} {a}", f"{t}의 {a}", f"{t} 관련 {a}", f"{t}에 대한 {a}"), "so")


def search_vp(r: R, t: str, a: str, form: str = "final") -> str:
    return f"{s_obj(r, t, a)}{eul(a)} {_form(r, SEARCH_V, form)}"


def search_implicit(r: R, t: str, a: str) -> str:
    """Retrieval request without any search verb."""
    return r.pick((f"요즘 {t} {a}{i_ga(a)} 어때?", f"{t} {a}{i_ga(a)} 어떻게 되는지 {r.end('알려')}",
                   f"{t} {a} 좀 {r.end('알려')}", f"{t}{eun(t)} {a}{i_ga(a)} 어떻게 돼?",
                   f"최근 {t} {a}{i_ga(a)} 궁금한데 {r.end('알려')}"), "si")


def sql_ctx(r: R, s: dict) -> str:
    return r.pick((f"{s['system']}에서", f"{s['owner']}", "DB에서", "사내 데이터에서", ""), "ctx")


def sql_vp(r: R, s: dict, q: str = "Q", form: str = "final") -> str:
    return f"{sql_ctx(r, s)} {s[q]}{eul(s[q])} {_form(r, SQL_V, form)}".strip()


def summary_vp(r: R, ref: str | None, form: str = "final") -> str:
    head = f"{ref}{eul(ref)} " if ref else ""
    return head + _form(r, SUMMARY_V, form)


def extract_vp(r: R, f: str, ref: str | None = None, form: str = "final") -> str:
    head = r.pick((f"{f}만", f"{f}{eul(f)}", f"{f} 부분만"), "ef")
    where = f"{ref}에서 " if ref else ""
    return f"{where}{head} {_form(r, EXTRACT_V, form)}"


def compare_vp(r: R, c: str, x: str | None = None, y: str | None = None, form: str = "final") -> str:
    if x and y:
        head = r.pick((f"{x}{wa(x)} {y}{eul(y)} {c} 기준으로", f"{x}{wa(x)} {y}의 {c}{eul(c)}",
                       f"{c} 면에서 {x}{wa(x)} {y}{eul(y)}"), "ch")
    else:
        head = r.pick((f"{c}{eul(c)}", f"{c} 기준으로", f"{c} 차이를"), "ch")
    return f"{head} {_form(r, COMPARE_V, form)}"


def embed(r: R, core: str, x: str, style: str) -> str:
    if style == "colon":
        return f"{core}: {x}"
    if style == "after":
        return f"{sent(core)}\n\n{x}"
    if style == "before":
        return f"{x}\n\n{core}"
    if style == "quote":
        return f"\"{x}\" {core}"
    if style == "label":
        return f"[본문]\n{x}\n[요청] {core}"
    raise ValueError(style)


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Taxonomy:
    name: str
    group: str  # normal | special | hard
    description: str
    difficulty: str
    pilot_quota: int
    hard_case: str | None = None


TAXONOMY: tuple[Taxonomy, ...] = (
    Taxonomy("single_search", "normal", "external retrieval only", "easy", 12),
    Taxonomy("single_summary_user", "normal", "summarize user-supplied content", "easy", 10),
    Taxonomy("single_extract_user", "normal", "extract named fields from user content", "easy", 10),
    Taxonomy("single_sql", "normal", "natural-language database request", "easy", 10),
    Taxonomy("single_compare_user", "normal", "compare targets inside one user document", "medium", 5),
    Taxonomy("search_summary", "normal", "retrieve, then summarize the results", "medium", 9),
    Taxonomy("search_extract", "normal", "retrieve, then extract fields from the results", "medium", 8),
    Taxonomy("sql_summary", "normal", "query, then summarize the results", "medium", 7),
    Taxonomy("sql_extract", "normal", "query, then extract fields from the results", "medium", 6),
    Taxonomy("extract_summary_user", "normal", "extract from user content, then summarize the extract", "medium", 5),
    Taxonomy("search_search_compare", "normal", "retrieve two targets, compare", "medium", 9),
    Taxonomy("sql_sql_compare", "normal", "two queries, compare", "medium", 6),
    Taxonomy("search_sql_compare", "normal", "retrieval vs internal data, compare", "hard", 6),
    Taxonomy("user_search_compare", "normal", "user content vs retrieved information", "hard", 4),
    Taxonomy("ssc_summary", "normal", "two retrievals, compare, summarize the comparison (4 tasks)", "hard", 5),
    Taxonomy("search_extract_summary", "normal", "retrieve, extract, summarize (3 tasks)", "hard", 5),
    Taxonomy("sql_extract_summary", "normal", "query, extract, summarize (3 tasks)", "hard", 4),
    Taxonomy("multi_output", "normal", "several final outputs", "hard", 8),
    Taxonomy("clarify", "special", "required information missing", "medium", 10),
    Taxonomy("unsupported", "special", "no registered skill can do it", "medium", 10),
    Taxonomy("hard_search_word_no_search", "hard", "검색 appears but no retrieval is needed", "hard", 8,
             "search_word_no_search"),
    Taxonomy("hard_summary_word_no_summary", "hard", "요약 appears but summary is not the task", "hard", 8,
             "summary_word_no_summary"),
    Taxonomy("hard_implicit_retrieval", "hard", "retrieval needed without any search verb", "hard", 8,
             "implicit_retrieval"),
    Taxonomy("hard_short_multi", "hard", "terse request needing 2-4 tasks", "hard", 7, "short_multi"),
    Taxonomy("hard_long_single", "hard", "long request needing one task", "hard", 7, "long_single"),
    Taxonomy("hard_order_trap", "hard", "surface order differs from execution order", "hard", 7, "order_trap"),
    Taxonomy("hard_unnecessary_skill", "hard", "extra skill is mentioned only to exclude it", "hard", 6,
             "unnecessary_skill"),
)
TAXONOMY_BY_NAME = {t.name: t for t in TAXONOMY}

# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Family:
    id: str
    taxonomy: str
    intent: str
    render: Callable[[R, dict], str]
    tier: str = "core"         # core: pilot + 5k, ext: 5k only
    stratum: str | None = None  # split assignment is stratified by (taxonomy, stratum)
    verbosity: str = "random"   # random | none | long


FAMILIES: list[Family] = []


def fam(id, taxonomy, intent, render, tier="core", stratum=None, verbosity="random"):
    FAMILIES.append(Family(id, taxonomy, intent, render, tier, stratum, verbosity))


# ---- single_search ------------------------------------------------------------
fam("ss_direct", "single_search", "search", lambda r, s: search_vp(r, s["T"], s["A"]))
fam("ss_noun", "single_search", "search",
    lambda r, s: f"{s_obj(r, s['T'], s['A'])} {_noun(r, SEARCH_V)} {r.req()}")
fam("ss_topic_last", "single_search", "search",
    lambda r, s: f"{sent(s['A'] + eul(s['A']) + ' ' + _form(r, SEARCH_V, 'final'))} 대상은 {s['T']}.")
fam("ss_curious", "single_search", "search",
    lambda r, s: f"{s['T']} {s['A']}{i_ga(s['A'])} 궁금해. {search_vp(r, s['T'], s['A'])}")
fam("ss_about", "single_search", "search",
    lambda r, s: f"{s['T']}에 대해 {s['A']}{eul(s['A'])} {_form(r, SEARCH_V, 'final')}")
fam("ss_web", "single_search", "search",
    lambda r, s: f"인터넷에서 {search_vp(r, s['T'], s['A'])}", tier="ext")
fam("ss_material", "single_search", "search",
    lambda r, s: f"{s['T']} {s['A']} 관련 자료 좀 {r.end(r.pick(('찾아', '모아', '구해'), 'm'))}", tier="ext")
fam("ss_latest_check", "single_search", "search",
    lambda r, s: f"{s['T']} {s['A']}, 최신 기준으로 {r.end(r.pick(('확인해', '찾아봐', '알아봐'), 'm'))}", tier="ext")

# ---- single_summary_user --------------------------------------------------------
fam("su_colon", "single_summary_user", "summary_user", lambda r, s: embed(r, summary_vp(r, r.uref()), s["X"], "colon"))
fam("su_after", "single_summary_user", "summary_user", lambda r, s: embed(r, summary_vp(r, r.uref()), s["X"], "after"))
fam("su_before", "single_summary_user", "summary_user",
    lambda r, s: embed(r, summary_vp(r, r.pick(("위 내용", "위 글", "이거"), "w")), s["X"], "before"))
fam("su_quote", "single_summary_user", "summary_user",
    lambda r, s: embed(r, f"이거 {_form(r, SUMMARY_V, 'final')}", s["X"], "quote"))
fam("su_noun", "single_summary_user", "summary_user",
    lambda r, s: embed(r, f"{r.pick(('한 줄 요약', '짧은 요약', '요약', '핵심 정리'), 'n')} {r.req()}", s["X"], "before"))
fam("su_what_about", "single_summary_user", "summary_user",
    lambda r, s: embed(r, f"이거 무슨 내용인지 짧게 {r.end('말해')}", s["X"], "before"), tier="ext")
fam("su_label", "single_summary_user", "summary_user",
    lambda r, s: embed(r, summary_vp(r, None), s["X"], "label"), tier="ext")
fam("su_gist", "single_summary_user", "summary_user",
    lambda r, s: embed(r, f"{r.uref()}의 요지가 뭔지 한두 문장으로 {r.end('알려')}", s["X"], "colon"), tier="ext")

# ---- single_extract_user ----------------------------------------------------------
fam("se_colon", "single_extract_user", "extract_user",
    lambda r, s: embed(r, extract_vp(r, s["F"], r.uref()), s["X"], "colon"))
fam("se_question", "single_extract_user", "extract_user",
    lambda r, s: embed(r, f"{r.uref()}에 나온 {s['F']}{i_ga(s['F'])} 뭐야?", s["X"], "after"))
fam("se_quote", "single_extract_user", "extract_user",
    lambda r, s: embed(r, f"여기서 {extract_vp(r, s['F'])}", s["X"], "quote"))
fam("se_before", "single_extract_user", "extract_user",
    lambda r, s: embed(r, extract_vp(r, s["F"], "위 내용"), s["X"], "before"))
fam("se_list", "single_extract_user", "extract_user",
    lambda r, s: embed(r, f"{s['F']}만 목록으로 {r.end('정리해')}", s["X"], "colon"))
fam("se_tell", "single_extract_user", "extract_user",
    lambda r, s: embed(r, f"여기 나온 {s['F']}만 {r.end('알려')}", s["X"], "before"), tier="ext")
fam("se_label", "single_extract_user", "extract_user",
    lambda r, s: embed(r, f"{s['F']} {_noun(r, EXTRACT_V)}", s["X"], "label"), tier="ext")
fam("se_check", "single_extract_user", "extract_user",
    lambda r, s: embed(r, f"{r.uref()}에서 {s['F']}에 해당하는 부분만 {r.end('표시해')}", s["X"], "colon"), tier="ext")

# ---- single_sql ----------------------------------------------------------------
fam("sq_direct", "single_sql", "sql", lambda r, s: sql_vp(r, s))
fam("sq_question", "single_sql", "sql",
    lambda r, s: f"{s['owner']} {s['Q']}{i_ga(s['Q'])} 어떻게 돼?")
fam("sq_noun", "single_sql", "sql", lambda r, s: f"{s['system']} 기준 {s['Q']} {_noun(r, SQL_V)} {r.req()}")
fam("sq_system_first", "single_sql", "sql",
    lambda r, s: f"{s['system']} 열어서 {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'final')}")
fam("sq_tell", "single_sql", "sql", lambda r, s: f"{s['owner']} 데이터 기준으로 {s['Q']} {r.end('알려')}")
fam("sq_report", "single_sql", "sql",
    lambda r, s: f"보고용으로 {s['Q']} 수치가 필요해. {sql_vp(r, s)}", tier="ext")
fam("sq_numbers", "single_sql", "sql",
    lambda r, s: f"{s['Q']}, 숫자로 {r.end('뽑아')}", tier="ext")
fam("sq_table", "single_sql", "sql",
    lambda r, s: f"{s['system']} 테이블 기준으로 {s['Q']} 좀 {r.end('확인해')}", tier="ext")

# ---- single_compare_user --------------------------------------------------------
fam("scu_colon", "single_compare_user", "compare_user",
    lambda r, s: embed(r, compare_vp(r, s["C"], s["TT1"], s["TT2"]), s["X"], "colon"))
fam("scu_after", "single_compare_user", "compare_user",
    lambda r, s: embed(r, f"아래 두 설명을 보고 {compare_vp(r, s['C'], s['TT1'], s['TT2'])}", s["X"], "after"))
fam("scu_question", "single_compare_user", "compare_user",
    lambda r, s: embed(r, f"{s['TT1']}{wa(s['TT1'])} {s['TT2']}의 {s['C']} 차이가 뭐야?", s["X"], "before"))
fam("scu_before", "single_compare_user", "compare_user",
    lambda r, s: embed(r, f"위 내용 기준으로 {compare_vp(r, s['C'], s['TT1'], s['TT2'])}", s["X"], "before"))
fam("scu_which", "single_compare_user", "compare_user",
    lambda r, s: embed(r, f"{s['TT1']}{wa(s['TT1'])} {s['TT2']} 중 {s['C']} 면에서 어떤 차이가 있는지 {r.end('정리해')}",
                       s["X"], "colon"))
fam("scu_label", "single_compare_user", "compare_user",
    lambda r, s: embed(r, f"{s['TT1']} vs {s['TT2']} {s['C']} 비교", s["X"], "label"), tier="ext")
fam("scu_quote", "single_compare_user", "compare_user",
    lambda r, s: embed(r, f"{s['TT1']}, {s['TT2']} 두 대상을 {s['C']} 기준으로 {_form(r, COMPARE_V, 'final')}", s["X"],
                       "quote"), tier="ext")

# ---- generic 2-step chain frames ---------------------------------------------------


def chain_families(tax: str, intent: str, first, second_bare, second_ref, first_noun, second_noun, prev_op,
                   payload_style: str | None = None, ext_first_implicit=None):
    """first(r,s,form) -> clause; second_bare(r,s) final clause without a reference;
    second_ref(r,s,ref) final clause with a reference to the first result."""
    wrap = (lambda core, r, s: embed(r, core, s["X"], payload_style)) if payload_style else (lambda core, r, s: core)
    fam(f"{tax}.go", tax, intent, lambda r, s: wrap(f"{first(r, s, 'go')} {second_bare(r, s)}", r, s))
    fam(f"{tax}.seo", tax, intent, lambda r, s: wrap(f"{first(r, s, 'seo')} {second_bare(r, s)}", r, s))
    fam(f"{tax}.then_ref", tax, intent,
        lambda r, s: wrap(f"{first(r, s, 'then')}, {second_ref(r, s, r.ref(prev_op))}", r, s))
    fam(f"{tax}.numbered", tax, intent,
        lambda r, s: wrap(f"순서대로 {sent(r.req())} 1) {first_noun(r, s)} 2) {second_noun(r, s)}", r, s))
    fam(f"{tax}.two_sentences", tax, intent,
        lambda r, s: wrap(f"{sent(first(r, s, 'final'))} {second_ref(r, s, r.pick(('그다음 그 결과', '그 결과', '이어서 그 결과'), 'nx'))}",
                          r, s))
    fam(f"{tax}.goal_first", tax, intent,
        lambda r, s: wrap(f"{second_noun(r, s)}{i_ga(second_noun(r, s))} 필요해. {first(r, s, 'seo')} {second_bare(r, s)}",
                          r, s), tier="ext")
    fam(f"{tax}.and_also", tax, intent,
        lambda r, s: wrap(f"{first(r, s, 'go')}, {r.pick(('이어서', '바로', '그걸로'), 'aa')} {second_bare(r, s)}", r, s),
        tier="ext")
    if ext_first_implicit:
        fam(f"{tax}.implicit", tax, intent,
            lambda r, s: wrap(f"{ext_first_implicit(r, s)} {second_ref(r, s, r.pick(('알아본 내용', '그 결과'), 'ir'))}", r, s),
            tier="ext")


def _sfirst(r, s, form):
    return search_vp(r, s["T"], s["A"], form)


def _sqlfirst(r, s, form):
    return sql_vp(r, s, "Q", form)


def _ufirst_extract(r, s, form):
    return extract_vp(r, s["F"], r.uref(), form)


chain_families("search_summary", "search_summary", _sfirst,
               lambda r, s: summary_vp(r, None), lambda r, s, ref: summary_vp(r, ref),
               lambda r, s: f"{s['T']} {s['A']} {_noun(r, SEARCH_V)}", lambda r, s: "결과 요약",
               "search", ext_first_implicit=lambda r, s: sent(f"{s['T']} {s['A']}{i_ga(s['A'])} 어떤지 좀 알아봐."))
chain_families("search_extract", "search_extract", _sfirst,
               lambda r, s: extract_vp(r, s["F"]), lambda r, s, ref: extract_vp(r, s["F"], ref),
               lambda r, s: f"{s['T']} {s['A']} {_noun(r, SEARCH_V)}", lambda r, s: f"{s['F']} 추출", "search",
               ext_first_implicit=lambda r, s: sent(f"{s['T']} {s['A']} 관련 소식을 좀 모아 봐"))
chain_families("sql_summary", "sql_summary", _sqlfirst,
               lambda r, s: summary_vp(r, None), lambda r, s, ref: summary_vp(r, ref),
               lambda r, s: f"{s['Q']} {_noun(r, SQL_V)}", lambda r, s: "결과 요약", "sql")
chain_families("sql_extract", "sql_extract", _sqlfirst,
               lambda r, s: extract_vp(r, s["F"]), lambda r, s, ref: extract_vp(r, s["F"], ref),
               lambda r, s: f"{s['Q']} {_noun(r, SQL_V)}", lambda r, s: f"{s['F']} 추출", "sql")
chain_families("extract_summary_user", "extract_summary_user", _ufirst_extract,
               lambda r, s: summary_vp(r, None), lambda r, s, ref: summary_vp(r, ref),
               lambda r, s: f"{s['F']} {_noun(r, EXTRACT_V)}", lambda r, s: "추출한 내용 요약", "extract",
               payload_style="colon")

# ---- search + search -> compare ------------------------------------------------------
fam("ssc.each", "search_search_compare", "search_search_compare",
    lambda r, s: f"{s['T1']}{wa(s['T1'])} {s['T2']}의 {s['A']}{eul(s['A'])} 각각 {_form(r, SEARCH_V, 'seo')} "
                 f"{_form(r, COMPARE_V, 'final')}")
fam("ssc.pair_seo", "search_search_compare", "search_search_compare",
    lambda r, s: f"{s['T1']}, {s['T2']} {s['A']} {_form(r, SEARCH_V, 'seo')} {_form(r, COMPARE_V, 'final')}")
fam("ssc.numbered", "search_search_compare", "search_search_compare",
    lambda r, s: f"순서대로 {sent(r.req())} 1) {s['T1']} {s['A']} {_noun(r, SEARCH_V)} 2) {s['T2']} {s['A']} "
                 f"{_noun(r, SEARCH_V)} 3) 두 결과 비교")
fam("ssc.need", "search_search_compare", "search_search_compare",
    lambda r, s: f"{s['C']} 비교가 필요해. {s['T1']}{wa(s['T1'])} {s['T2']} 자료를 {_form(r, SEARCH_V, 'seo')} "
                 f"{r.end(r.pick(('비교해', '대조해'), 'cv'))}")
fam("ssc.which", "search_search_compare", "search_search_compare",
    lambda r, s: f"{s['T1']}{wa(s['T1'])} {s['T2']} 중 {s['C']} 면에서 어느 쪽이 나은지 "
                 f"{_form(r, SEARCH_V, 'go')} {r.end('비교해')}")
fam("ssc.versus", "search_search_compare", "search_search_compare",
    lambda r, s: f"{s['T1']} 대 {s['T2']}, {s['C']} {_form(r, SEARCH_V, 'seo')} {r.end('비교해')}",
    tier="ext")
fam("ssc.two_sentences", "search_search_compare", "search_search_compare",
    lambda r, s: f"{sent(search_vp(r, s['T1'], s['A']))} {s['T2']}도 {r.pick(('똑같이', '같은 식으로'), 'dd')} "
                 f"{sent(r.end(r.pick(('찾아', '알아봐'), 'm')))} 그리고 두 결과를 {_form(r, COMPARE_V, 'final')}", tier="ext")
fam("ssc.question", "search_search_compare", "search_search_compare",
    lambda r, s: f"{s['T1']}{wa(s['T1'])} {s['T2']}의 {s['C']}, 찾아서 비교하면 어때? {r.req()}", tier="ext")

# ---- sql + sql -> compare --------------------------------------------------------------
fam("qqc.each", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"{s['system']}에서 {s['Q1']}{wa(s['Q1'])} {s['Q2']}{eul(s['Q2'])} 각각 {_form(r, SQL_V, 'seo')} "
                 f"{r.end('비교해')}")
fam("qqc.numbered", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"1) {s['Q1']} {_noun(r, SQL_V)} 2) {s['Q2']} {_noun(r, SQL_V)} 3) {s['C']} 비교. {r.req()}")
fam("qqc.diff", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"{s['owner']} {s['C']} 차이가 궁금해. {s['Q1']}{wa(s['Q1'])} {s['Q2']}{eul(s['Q2'])} "
                 f"{_form(r, SQL_V, 'go')} {r.end('비교해')}")
fam("qqc.two_sentences", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"{sent(sql_vp(r, s, 'Q1'))} {sent(s['Q2'] + '도 ' + _form(r, SQL_V, 'final'))} "
                 f"두 결과를 {r.end(r.pick(('비교해', '견줘 봐'), 'cv'))}")
fam("qqc.against", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"{s['Q1']}{wa(s['Q1'])} {s['Q2']}, DB에서 {_form(r, SQL_V, 'seo')} {s['C']} 기준으로 {r.end('대조해')}")
fam("qqc.change", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"{s['C']}{i_ga(s['C'])} 어떻게 달라졌는지 {s['system']} 데이터로 {r.end('비교해')} "
                 f"({s['Q1']} vs {s['Q2']})", tier="ext")
fam("qqc.need", "sql_sql_compare", "sql_sql_compare",
    lambda r, s: f"보고서에 {s['C']} 비교를 넣어야 해. {s['Q1']}{wa(s['Q1'])} {s['Q2']} 조회해서 {r.end('비교해')}",
    tier="ext")

# ---- search + sql -> compare -------------------------------------------------------------


def _ssq_mentions(r, s):
    ext = f"{s['E']} {s['AE']}"
    internal = s["QI"]
    return (internal, ext) if s.get("order") == "sql_first" else (ext, internal)


fam("sqc.and", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{a}{wa(a)} {b}{eul(b)} {r.end('비교해')}")(*_ssq_mentions(r, s)))
fam("sqc.fetch_both", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{a}, {b} 둘 다 {r.pick(('확인해서', '알아봐서', '찾아서'), 'fb')} "
                               f"{s['CI']} {r.end('비교해')}")(*_ssq_mentions(r, s)))
fam("sqc.numbered", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"1) {a} 확인 2) {b} 확인 3) 비교. {r.req()}")(*_ssq_mentions(r, s)))
fam("sqc.gap", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{a}{wa(a)} {b} 사이에 차이가 얼마나 나는지 {r.end('따져 봐')}")(*_ssq_mentions(r, s)))
fam("sqc.check_vs", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{a}{i_ga(a)} {b}{wa(b)} 비교해서 어떤지 {r.end('알려')}")(*_ssq_mentions(r, s)))
fam("sqc.against", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{a} 대비 {b}, 한번 {r.end('비교해')}")(*_ssq_mentions(r, s)), tier="ext")
fam("sqc.need", "search_sql_compare", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{s['CI']} 비교 자료가 필요해요. {a}{wa(a)} {b}{eul(b)} {_form(r, SEARCH_V, 'go')} "
                               f"{r.end('비교해')}")(*_ssq_mentions(r, s)), tier="ext")

# ---- user + search -> compare --------------------------------------------------------------
fam("usc.check", "user_search_compare", "user_search_compare",
    lambda r, s: embed(r, f"이 글 내용이 요즘 {s['T']} 정보랑 맞는지 최신 자료를 {_form(r, SEARCH_V, 'seo')} {r.end('비교해')}",
                       s["X"], "colon"))
fam("usc.after", "user_search_compare", "user_search_compare",
    lambda r, s: embed(r, f"{s['T']} 최신 정보를 {_form(r, SEARCH_V, 'go')} 위 글과 {r.end('대조해')}", s["X"], "before"))
fam("usc.outdated", "user_search_compare", "user_search_compare",
    lambda r, s: (lambda u: embed(r, f"{u}{i_ga(u)} 오래된 것 같아. {s['T']} 최신 정보를 {_form(r, SEARCH_V, 'seo')} "
                                     f"달라진 점을 {r.end('비교해')}", s["X"], "after"))(r.uref()))
fam("usc.quote", "user_search_compare", "user_search_compare",
    lambda r, s: embed(r, f"이 설명을 {s['T']} 관련 최신 자료와 {r.end('비교해')}", s["X"], "quote"))
fam("usc.label", "user_search_compare", "user_search_compare",
    lambda r, s: embed(r, f"{s['T']} 최신 정보 {_noun(r, SEARCH_V)} 후 본문과 비교", s["X"], "label"))
fam("usc.question", "user_search_compare", "user_search_compare",
    lambda r, s: embed(r, f"{s['T']}에 대해 지금 나와 있는 정보랑 비교하면 뭐가 달라?", s["X"], "before"), tier="ext")

# ---- 3+ task plans --------------------------------------------------------------------------
fam("s4.go", "ssc_summary", "ssc_summary",
    lambda r, s: f"{s['T1']}{wa(s['T1'])} {s['T2']}의 {s['A']}{eul(s['A'])} {_form(r, SEARCH_V, 'seo')} "
                 f"비교하고, {r.pick(('비교 결과', '그 결과'), 'x')}{eul('결과')} {_form(r, SUMMARY_V, 'final')}")
fam("s4.numbered", "ssc_summary", "ssc_summary",
    lambda r, s: f"1) {s['T1']} {s['A']} 검색 2) {s['T2']} {s['A']} 검색 3) 두 결과 비교 4) 비교 결과 요약. {r.req()}")
fam("s4.then", "ssc_summary", "ssc_summary",
    lambda r, s: f"{s['T1']}{wa(s['T1'])} {s['T2']} 자료를 각각 {_form(r, SEARCH_V, 'then')} {s['C']}{eul(s['C'])} "
                 f"비교하고, 마지막에 한 문단으로 {r.end('정리해')}")
fam("s4.goal", "ssc_summary", "ssc_summary",
    lambda r, s: f"{s['T1']} vs {s['T2']} 비교 요약본이 필요해. {s['A']} {_form(r, SEARCH_V, 'seo')} 비교한 다음 "
                 f"{r.end('간추려')}")
fam("s4.two_sentences", "ssc_summary", "ssc_summary",
    lambda r, s: f"{sent(s['T1'] + wa(s['T1']) + ' ' + s['T2'] + ' ' + s['A'] + eul(s['A']) + ' ' + _form(r, SEARCH_V, 'go') + ' 차이를 비교해')} "
                 f"비교한 내용은 짧게 {r.end('요약해')}")
fam("s4.brief", "ssc_summary", "ssc_summary",
    lambda r, s: f"{s['T1']}하고 {s['T2']} {s['A']} 찾아서 비교한 걸 요약본으로 {r.end('만들어')}", tier="ext")

fam("s3.go", "search_extract_summary", "search_extract_summary",
    lambda r, s: f"{search_vp(r, s['T'], s['A'], 'seo')} {extract_vp(r, s['F'], None, 'then')} {summary_vp(r, None)}")
fam("s3.numbered", "search_extract_summary", "search_extract_summary",
    lambda r, s: f"1) {s['T']} {s['A']} 검색 2) {s['F']} 추출 3) 추출한 내용 요약. {r.req()}")
fam("s3.then", "search_extract_summary", "search_extract_summary",
    lambda r, s: f"{search_vp(r, s['T'], s['A'], 'then')}, 거기서 {s['F']}만 {_form(r, EXTRACT_V, 'go')} "
                 f"그걸 {_form(r, SUMMARY_V, 'final')}")
fam("s3.goal", "search_extract_summary", "search_extract_summary",
    lambda r, s: f"{s['T']} {s['F']} 정리본이 필요해. {s['A']}{eul(s['A'])} {_form(r, SEARCH_V, 'seo')} {s['F']}만 "
                 f"{_form(r, EXTRACT_V, 'go')} {r.end('요약해')}")
fam("s3.sentences", "search_extract_summary", "search_extract_summary",
    lambda r, s: f"{sent(search_vp(r, s['T'], s['A']))} {sent('그중 ' + extract_vp(r, s['F']))} "
                 f"마지막으로 뽑은 걸 {r.end('간추려')}")
fam("s3.brief", "search_extract_summary", "search_extract_summary",
    lambda r, s: f"{s['T']} {s['A']} 찾아서 {s['F']}만 뽑고 요약까지 {r.req()}", tier="ext")

fam("q3.go", "sql_extract_summary", "sql_extract_summary",
    lambda r, s: f"{sql_vp(r, s, 'Q', 'seo')} {extract_vp(r, s['F'], None, 'then')} {summary_vp(r, None)}")
fam("q3.numbered", "sql_extract_summary", "sql_extract_summary",
    lambda r, s: f"1) {s['Q']} 조회 2) {s['F']} 추출 3) 요약. {r.req()}")
fam("q3.then", "sql_extract_summary", "sql_extract_summary",
    lambda r, s: f"{sql_vp(r, s, 'Q', 'then')}, 결과에서 {s['F']}만 {_form(r, EXTRACT_V, 'go')} 짧게 {r.end('정리해')}")
fam("q3.goal", "sql_extract_summary", "sql_extract_summary",
    lambda r, s: f"주간 보고에 넣을 요약이 필요해. {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'seo')} {s['F']}만 추린 다음 "
                 f"{r.end('요약해')}")
fam("q3.sentences", "sql_extract_summary", "sql_extract_summary",
    lambda r, s: f"{sent(sql_vp(r, s))} {sent('거기서 ' + extract_vp(r, s['F']))} 뽑은 내용을 한 문단으로 {r.end('정리해')}")
fam("q3.brief", "sql_extract_summary", "sql_extract_summary",
    lambda r, s: f"{s['Q']} 조회해서 {s['F']}만 뽑아 요약해 {r.pick(('줘', '주세요', '줄래?'), 'z')}", tier="ext")

# ---- multiple outputs -------------------------------------------------------------------------
fam("mo.search_both", "multi_output", "mo_search_summary_extract",
    lambda r, s: f"{search_vp(r, s['T'], s['A'], 'seo')} 요약해 주고, {s['F']}{eun(s['F'])} 따로 {r.end('뽑아')}",
    stratum="search_summary_extract")
fam("mo.search_two_results", "multi_output", "mo_search_summary_extract",
    lambda r, s: f"{s['T']} {s['A']} 찾아서 두 가지로 {sent(r.req())} 요약본 하나, {s['F']} 목록 하나.",
    stratum="search_summary_extract")
fam("mo.search_separate", "multi_output", "mo_search_summary_extract",
    lambda r, s: f"{sent(search_vp(r, s['T'], s['A']))} 결과는 요약본이랑 {s['F']} 정리본을 따로따로 줘.",
    stratum="search_summary_extract", tier="ext")
fam("mo.user_both", "multi_output", "mo_user_summary_extract",
    lambda r, s: embed(r, f"{r.uref()} 요약이랑, {s['F']}만 따로 {r.end('정리해')}", s["X"], "colon"),
    stratum="user_summary_extract")
fam("mo.user_two", "multi_output", "mo_user_summary_extract",
    lambda r, s: embed(r, f"위 글로 두 가지 {sent(r.req())} 요약하고, {s['F']}도 따로 뽑아서 둘 다 보여 줘.", s["X"], "before"),
    stratum="user_summary_extract")
fam("mo.user_label", "multi_output", "mo_user_summary_extract",
    lambda r, s: embed(r, f"요약 + {s['F']} 추출, 결과 둘 다", s["X"], "label"), stratum="user_summary_extract", tier="ext")
fam("mo.search_sql", "multi_output", "mo_search_sql",
    lambda r, s: (lambda a, b: f"{a} {r.pick(('알아봐 주고', '찾아 주고'), 'k')}, {b}도 따로 {r.end('알려')}")(*_ssq_mentions(r, s)),
    stratum="search_sql")
fam("mo.search_sql_list", "multi_output", "mo_search_sql",
    lambda r, s: (lambda a, b: f"두 가지 {sent(r.req())} {a}, 그리고 {b}. 비교는 필요 없고 각각 보여 줘.")(*_ssq_mentions(r, s)),
    stratum="search_sql")
fam("mo.search_search", "multi_output", "mo_search_search",
    lambda r, s: f"{search_vp(r, s['T1'], s['A'], 'go')}, {s['T2']} {s['A2']}도 따로 {r.end('찾아')}",
    stratum="search_search")
fam("mo.search_search_list", "multi_output", "mo_search_search",
    lambda r, s: f"각각 {sent(r.end('찾아'))} {s['T1']} {s['A']}, {s['T2']} {s['A2']}. 결과는 따로 보여 줘.",
    stratum="search_search", tier="ext")

# ---- clarify (stratum = reason) ------------------------------------------------------------------
for reason, pairs in {
    "missing_payload_summary": (("cl.sum_ref", lambda r, s: summary_vp(r, r.uref())),
                                ("cl.sum_noun", lambda r, s: f"{r.uref()} 요약 {r.req()}"),
                                ("cl.sum_this", lambda r, s: f"이거 {_form(r, SUMMARY_V, 'final')}")),
    "missing_payload_extract": (("cl.ext_ref", lambda r, s: extract_vp(r, s["F"], r.uref())),
                                ("cl.ext_question", lambda r, s: f"{r.uref()}에 나온 {s['F']}{i_ga(s['F'])} 뭐야?"),
                                ("cl.ext_list", lambda r, s: f"여기서 {s['F']}만 목록으로 {r.end('정리해')}")),
    "missing_payload_compare": (("cl.cmp_docs", lambda r, s: f"이 두 문서 {r.end('비교해')}"),
                                ("cl.cmp_below", lambda r, s: f"아래 자료들 {s['C']} 기준으로 {r.end('비교해')}"),
                                ("cl.cmp_attached", lambda r, s: f"붙여 준 두 글 차이 좀 {r.end('알려')}")),
    "missing_referent": (("cl.ref_find", lambda r, s: f"{s['P']} {s['A']} 좀 {_form(r, SEARCH_V, 'final')}"),
                         ("cl.ref_question", lambda r, s: f"{s['P']} {s['A']}{i_ga(s['A'])} 어떻게 돼?"),
                         ("cl.ref_again", lambda r, s: f"아까 {s['P']} 다시 {r.end('찾아봐')}")),
    "missing_compare_target": (("cl.one_target", lambda r, s: f"{s['T']}{eul(s['T'])} {s['C']} 기준으로 {r.end('비교해')}"),
                               ("cl.one_target_q", lambda r, s: f"{s['T']}{wa(s['T'])} 비교하면 뭐가 나아?"),
                               ("cl.one_target_find", lambda r, s: f"{s['T']} {s['A']} 찾아서 {r.end('비교해')}")),
    "ambiguous_db": (("cl.db_vague", lambda r, s: f"{s['V']} {_form(r, SQL_V, 'final')}"),
                     ("cl.db_system", lambda r, s: f"DB에서 {s['V']} 좀 {r.end('뽑아')}"),
                     ("cl.db_question", lambda r, s: f"{s['V']} 어떻게 나왔어? {r.req()}")),
}.items():
    for family_id, render in pairs:
        fam(family_id, "clarify", "clarify", render, stratum=reason)
fam("cl.sum_short", "clarify", "clarify", lambda r, s: f"요약 {r.req()}", stratum="missing_payload_summary", tier="ext")
fam("cl.ext_short", "clarify", "clarify", lambda r, s: f"{s['F']}만 {r.end('뽑아')}", stratum="missing_payload_extract",
    tier="ext")
fam("cl.cmp_short", "clarify", "clarify", lambda r, s: f"둘이 비교 좀 {r.req()}", stratum="missing_payload_compare", tier="ext")
fam("cl.ref_it", "clarify", "clarify", lambda r, s: f"{s['P']} 최신 소식 {r.end('알려')}", stratum="missing_referent",
    tier="ext")
fam("cl.one_target_vs", "clarify", "clarify", lambda r, s: f"{s['T']} vs ? {s['C']} 비교", stratum="missing_compare_target",
    tier="ext")
fam("cl.db_numbers", "clarify", "clarify", lambda r, s: f"{s['V']} 조회 좀", stratum="ambiguous_db", tier="ext")

# ---- unsupported (content = category item; families = frames) -------------------------------------


def _unsupported_clause(r: R, s: dict) -> str:
    c, item = s["category"], s["item"]
    if c == "translate":
        return f"다음 문장을 {item}로 {r.end('번역해')}: {s['X']}"
    if c == "code":
        return f"{item} 코드를 {r.end('짜')}"
    if c == "image":
        return f"{item}{eul(item)} {r.end('그려')}"
    if c == "send":
        return f"{item}{eul(item)} {r.end('보내')}"
    if c == "calc":
        return f"{item}{eul(item)} {r.end('계산해')}"
    return f"{item} {r.req()}"  # file, schedule, booking, device, media


fam("un.direct", "unsupported", "unsupported", lambda r, s: _unsupported_clause(r, s))
fam("un.polite_can", "unsupported", "unsupported",
    lambda r, s: f"혹시 이런 것도 돼? {_unsupported_clause(r, s)}")
fam("un.need", "unsupported", "unsupported", lambda r, s: f"지금 바로 필요해서 그래. {_unsupported_clause(r, s)}")
fam("un.after_payload", "unsupported", "unsupported",
    lambda r, s: f"{sent(_unsupported_clause(r, s))} 끝나면 알려 줘.")
fam("un.then", "unsupported", "unsupported", lambda r, s: f"{sent(_unsupported_clause(r, s))} 가능하면 바로 처리해 줘.")
fam("un.question", "unsupported", "unsupported",
    lambda r, s: f"{sent(_unsupported_clause(r, s))} 이런 것도 할 수 있어?", tier="ext")
fam("un.context", "unsupported", "unsupported",
    lambda r, s: f"회의 끝나고 처리할 게 있어. {_unsupported_clause(r, s)}", tier="ext")

# ---- hard: 검색 word, no retrieval -------------------------------------------------------------------
fam("hsw.pasted_result", "hard_search_word_no_search", "summary_user",
    lambda r, s: embed(r, f"검색해서 찾은 글인데 너무 길어. {summary_vp(r, None)}", s["X"], "colon"), stratum="summary_user")
fam("hsw.pasted_extract", "hard_search_word_no_search", "extract_user",
    lambda r, s: embed(r, f"검색으로 찾아 둔 자료야. 여기서 {extract_vp(r, s['F'])}", s["X"], "after"), stratum="extract_user")
fam("hsw.searchlog_sql", "hard_search_word_no_search", "sql",
    lambda r, s: f"인터넷 검색 말고 {s['system']}에서 바로 {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'final')}", stratum="sql")
fam("hsw.no_search_needed", "hard_search_word_no_search", "summary_user",
    lambda r, s: embed(r, f"따로 검색할 필요 없이 {r.uref()}만 {_form(r, SUMMARY_V, 'final')}", s["X"], "colon"),
    stratum="summary_user")
fam("hsw.search_feature_sql", "hard_search_word_no_search", "sql",
    lambda r, s: f"검색 기능 개선 회의용이야. {sql_vp(r, s)}", stratum="sql")
fam("hsw.search_term_extract", "hard_search_word_no_search", "extract_user",
    lambda r, s: embed(r, f"검색 키워드로 쓸 {s['F']}만 {r.end('뽑아')}", s["X"], "colon"), stratum="extract_user")
fam("hsw.searched_compare", "hard_search_word_no_search", "compare_user",
    lambda r, s: embed(r, f"검색해서 모은 두 설명이야. {compare_vp(r, s['C'], s['TT1'], s['TT2'])}", s["X"], "after"),
    stratum="compare_user", tier="ext")
fam("hsw.searchlog_summary", "hard_search_word_no_search", "sql_summary",
    lambda r, s: f"검색은 필요 없고 사내 DB에서 {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'seo')} {summary_vp(r, None)}",
    stratum="sql_summary", tier="ext")

# ---- hard: 요약 word, no summary ---------------------------------------------------------------------
fam("hsm.extract_from_summary", "hard_summary_word_no_summary", "extract_user",
    lambda r, s: embed(r, f"아래 요약본에서 {extract_vp(r, s['F'])}", s["X"], "colon"), stratum="extract_user")
fam("hsm.find_summary_doc", "hard_summary_word_no_summary", "search_summary_doc",
    lambda r, s: f"{s['T']} {s['A']} 요약 자료 좀 {_form(r, SEARCH_V, 'final')}", stratum="search")
fam("hsm.sql_for_summary", "hard_summary_word_no_summary", "sql",
    lambda r, s: f"요약 보고서에 넣을 {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'final')}", stratum="sql")
fam("hsm.not_summary_compare", "hard_summary_word_no_summary", "search_search_compare",
    lambda r, s: f"요약 말고 {s['T1']}{wa(s['T1'])} {s['T2']}의 {s['A']}{eul(s['A'])} {_form(r, SEARCH_V, 'seo')} "
                 f"{_form(r, COMPARE_V, 'final')}", stratum="compare")
fam("hsm.summary_table_sql", "hard_summary_word_no_summary", "sql",
    lambda r, s: f"{s['system']} 요약 테이블에서 {s['Q']} {r.end('가져와')}", stratum="sql")
fam("hsm.already_summarized", "hard_summary_word_no_summary", "extract_user",
    lambda r, s: embed(r, f"요약은 이미 돼 있어. {s['F']}만 {r.end('골라')}", s["X"], "after"), stratum="extract_user")
fam("hsm.summary_not_needed_search", "hard_summary_word_no_summary", "search",
    lambda r, s: f"요약은 필요 없어. {search_vp(r, s['T'], s['A'])}", stratum="search", tier="ext")
fam("hsm.summary_of_sql", "hard_summary_word_no_summary", "sql_extract",
    lambda r, s: f"요약 지표 {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'seo')} {s['F']}만 {r.end('뽑아')}",
    stratum="sql_extract", tier="ext")

# ---- hard: implicit retrieval ---------------------------------------------------------------------
fam("hir.how_is", "hard_implicit_retrieval", "search", lambda r, s: f"요즘 {s['T']} {s['A']}{i_ga(s['A'])} 어때?",
    stratum="search")
fam("hir.what_is", "hard_implicit_retrieval", "search",
    lambda r, s: f"{s['T']}{eun(s['T'])} {s['A']}{i_ga(s['A'])} 어떻게 돼?", stratum="search")
fam("hir.tell", "hard_implicit_retrieval", "search", lambda r, s: f"{s['T']} {s['A']} 좀 {r.end('알려')}",
    stratum="search")
fam("hir.one_line", "hard_implicit_retrieval", "search_summary",
    lambda r, s: f"{s['T']} {s['A']}{i_ga(s['A'])} 요즘 어떤지 한 줄로 {r.end('말해')}", stratum="search_summary")
fam("hir.which_better", "hard_implicit_retrieval", "search_search_compare",
    lambda r, s: f"{s['T1']}하고 {s['T2']} 중에 {s['C']} 면에서 뭐가 나아?", stratum="compare")
fam("hir.numbers_only", "hard_implicit_retrieval", "search_extract",
    lambda r, s: f"요즘 {s['T']} {s['A']} 중에 {s['F']}만 딱 {r.end('알려')}", stratum="search_extract")
fam("hir.wonder", "hard_implicit_retrieval", "search",
    lambda r, s: f"{s['T']} {s['A']}{i_ga(s['A'])} 갑자기 궁금하네.", stratum="search", tier="ext")
fam("hir.brief_trend", "hard_implicit_retrieval", "search_summary",
    lambda r, s: f"{s['T']} {s['A']}, 요새 분위기 짧게 {r.end('전해')}", stratum="search_summary", tier="ext")

# ---- hard: short multi ------------------------------------------------------------------------------
fam("hsh.vs", "hard_short_multi", "search_search_compare",
    lambda r, s: f"{s['T1']} vs {s['T2']} {s['C']}", stratum="compare", verbosity="none")
fam("hsh.find_sum", "hard_short_multi", "search_summary",
    lambda r, s: f"{s['T']} {s['A']} 찾아서 요약", stratum="search_summary", verbosity="none")
fam("hsh.sql_sum", "hard_short_multi", "sql_summary",
    lambda r, s: f"{s['Q']} 조회 후 요약", stratum="sql_summary", verbosity="none")
fam("hsh.find_extract", "hard_short_multi", "search_extract",
    lambda r, s: f"{s['T']} {s['A']}, {s['F']}만", stratum="search_extract", verbosity="none")
fam("hsh.four", "hard_short_multi", "ssc_summary",
    lambda r, s: f"{s['T1']}·{s['T2']} {s['A']} 비교 후 한 줄 요약", stratum="ssc_summary", verbosity="none")
fam("hsh.dual", "hard_short_multi", "mo_user_summary_extract",
    lambda r, s: embed(r, f"요약+{s['F']}", s["X"], "colon"), stratum="multi_output", verbosity="none")
fam("hsh.sql_pair", "hard_short_multi", "sql_sql_compare",
    lambda r, s: f"{s['Q1']} vs {s['Q2']}", stratum="sql_compare", verbosity="none", tier="ext")
fam("hsh.web_vs_db", "hard_short_multi", "search_sql_compare",
    lambda r, s: (lambda a, b: f"{a} vs {b}")(*_ssq_mentions(r, s)), stratum="search_sql_compare", verbosity="none",
    tier="ext")

# ---- hard: long single --------------------------------------------------------------------------------
fam("hls.search", "hard_long_single", "search",
    lambda r, s: f"{s['LC']} 그래서 말인데, {search_vp(r, s['T'], s['A'])} 다른 건 안 해도 돼.", stratum="search",
    verbosity="none")
fam("hls.summary", "hard_long_single", "summary_user",
    lambda r, s: embed(r, f"{s['LC']} 아래 글을 다 읽을 시간이 없어서 그런데, {summary_vp(r, None)}", s["X"], "after"),
    stratum="summary_user", verbosity="none")
fam("hls.sql", "hard_long_single", "sql",
    lambda r, s: f"{s['LC']} 필요한 건 딱 하나야. {s['system']}에서 {s['Q']}{eul(s['Q'])} {_form(r, SQL_V, 'final')}",
    stratum="sql", verbosity="none")
fam("hls.extract", "hard_long_single", "extract_user",
    lambda r, s: embed(r, f"{s['LC']} 아래 자료에서 {extract_vp(r, s['F'])}", s["X"], "after"),
    stratum="extract_user", verbosity="none")
fam("hls.search_detail", "hard_long_single", "search",
    lambda r, s: f"{s['LC']} {s['T']}{eun(s['T'])} 이름만 들어 봤고 {s['A']}{eun(s['A'])} 전혀 몰라. 그 부분만 "
                 f"{r.end(r.pick(('찾아', '알아봐'), 'm'))}", stratum="search", verbosity="none")
fam("hls.sql_detail", "hard_long_single", "sql",
    lambda r, s: f"{s['LC']} 숫자만 있으면 나머지는 내가 정리할게. {s['Q']} {r.end('조회해')}", stratum="sql",
    verbosity="none", tier="ext")
fam("hls.compare_user", "hard_long_single", "compare_user",
    lambda r, s: embed(r, f"{s['LC']} 아래 {s['TT1']}{wa(s['TT1'])} {s['TT2']} 설명을 {s['C']} 기준으로 {r.end('비교해')}",
                       s["X"], "after"),
    stratum="compare_user", verbosity="none", tier="ext")

# ---- hard: order trap ----------------------------------------------------------------------------------
fam("hot.summary_before", "hard_order_trap", "search_summary",
    lambda r, s: f"{sent(_form(r, SUMMARY_V, 'final'))} 그 전에 {s['T']} {s['A']}부터 {r.end('찾아')}",
    stratum="search_summary")
fam("hot.compare_last", "hard_order_trap", "search_search_compare",
    lambda r, s: f"비교는 마지막에 {sent(r.end('해'))} 먼저 {s['T1']} {s['A']}, 그다음 {s['T2']} {s['A']}{eul(s['A'])} "
                 f"{r.end('찾아')}", stratum="compare")
fam("hot.extract_after_sql", "hard_order_trap", "sql_extract",
    lambda r, s: f"{sent(s['F'] + '만 ' + r.end('뽑아'))} 대상은 {s['Q']} 조회 결과야.", stratum="sql_extract")
fam("hot.summary_of_extract", "hard_order_trap", "extract_summary_user",
    lambda r, s: embed(r, f"요약해 줘. 단 전체가 아니라 {s['F']}만 먼저 뽑아서 그걸 요약해.", s["X"], "colon"),
    stratum="extract_summary_user")
fam("hot.goal_then_source", "hard_order_trap", "sql_summary",
    lambda r, s: f"요약본이 필요한데, 요약할 데이터는 {s['Q']}야. DB에서 먼저 {r.end('가져와')}", stratum="sql_summary")
fam("hot.extract_before", "hard_order_trap", "search_extract",
    lambda r, s: f"{s['F']}만 {sent(r.end('정리해'))} 정리하기 전에 {s['T']} {s['A']}{eul(s['A'])} 먼저 {r.end('찾아봐')}",
    stratum="search_extract", tier="ext")
fam("hot.summary_last", "hard_order_trap", "search_extract_summary",
    lambda r, s: f"마지막에 요약해 줘. 그 전에 {s['T']} {s['A']} 찾고, 거기서 {s['F']}만 {r.end('뽑아')}",
    stratum="search_extract_summary", tier="ext")

# ---- hard: unnecessary skill ---------------------------------------------------------------------------
fam("hun.summary_only", "hard_unnecessary_skill", "summary_user",
    lambda r, s: embed(r, f"{r.uref()} 요약만 해 줘. 검색은 필요 없어.", s["X"], "colon"), stratum="summary_user")
fam("hun.sql_only", "hard_unnecessary_skill", "sql",
    lambda r, s: f"{s['Q']} 조회만 {sent(r.req())} 요약은 안 해도 돼.", stratum="sql")
fam("hun.search_only", "hard_unnecessary_skill", "search",
    lambda r, s: f"{s['T']} {s['A']} 찾기만 해 줘. 정리는 내가 할게.", stratum="search")
fam("hun.extract_only", "hard_unnecessary_skill", "extract_user",
    lambda r, s: embed(r, f"{s['F']}만 {sent(r.end('뽑아'))} 요약이나 비교는 필요 없어.", s["X"], "colon"),
    stratum="extract_user")
fam("hun.compare_only", "hard_unnecessary_skill", "search_search_compare",
    lambda r, s: f"{s['T1']}{wa(s['T1'])} {s['T2']} {s['A']} 찾아서 비교만 해 줘. 따로 요약은 하지 마.", stratum="compare")
fam("hun.no_db", "hard_unnecessary_skill", "search",
    lambda r, s: f"사내 DB 말고 외부 자료로 {search_vp(r, s['T'], s['A'])}", stratum="search", tier="ext")
fam("hun.no_compare", "hard_unnecessary_skill", "search_summary",
    lambda r, s: f"비교는 필요 없고 {s['T']} {s['A']}만 찾아서 {r.end('정리해')}", stratum="search_summary", tier="ext")

FAMILY_BY_ID = {f.id: f for f in FAMILIES}
assert len(FAMILY_BY_ID) == len(FAMILIES), "duplicate family id"


def families_for(taxonomy: str, tiers: tuple[str, ...]) -> list[Family]:
    return [f for f in FAMILIES if f.taxonomy == taxonomy and f.tier in tiers]


def referenced_skills() -> set[str]:
    """Skill names used by canonical plans (checked against the Runtime registry)."""
    dummy = {"T": "t", "A": "a", "F": "f", "Q": "q", "T1": "t1", "T2": "t2", "A2": "a2", "C": "c", "Q1": "q1",
             "Q2": "q2", "E": "e", "AE": "ae", "QI": "qi", "CI": "ci", "TT1": "x", "TT2": "y",
             "reason": "missing_referent"}
    skills = set()
    for intent in INTENTS.values():
        skills.update(task["skill"] for task in intent(dummy)["tasks"])
    return skills
