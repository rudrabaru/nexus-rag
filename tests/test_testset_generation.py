"""Sampling, quality, prompt, parsing and generation of synthetic test questions."""
import json
import pytest
from src.evaluation.dataset import SYNTHETIC
from src.llm.client import LLMCall
from src.llm.errors import GenerationError
from src.llm.structured import extract_json_object
from src.testsets.generator import MAX_CONSECUTIVE_FAILURES, GenerationAborted, generate
from src.stores.testsets import PENDING
from src.testsets.models import Draft, DraftItem
from src.testsets.prompt import DIFFICULTY_INSTRUCTIONS, build_prompt, parse_generated
from src.testsets.quality import lexical_overlap, overlap_by_difficulty, tier_warnings
from src.testsets.sampling import ChunkGroup, SourceChunk, group_identical, interleave_by_document


def source_chunk(chunk_id, doc_id="d1", text=None, source=None, heading=("Setup",), code=False, table=False):
    return SourceChunk(
        chunk_id=chunk_id, doc_id=doc_id, source=source or f"https://example.com/{doc_id}", section_title=heading[-1] if heading else "",
        heading_path=tuple(heading), text=text or f"text of {chunk_id}", contains_code=code, contains_table=table,
    )


def group(chunk_id, **kwargs) -> ChunkGroup:
    return ChunkGroup((source_chunk(chunk_id, **kwargs),))


FAILURE = object()  # a scripted reply that makes the call raise, as the real client does once retries are spent


class FakeClient:
    """Answers each call from a script of replies (a dict becomes JSON); records the prompts it saw."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def call_llm(self, prompt, is_fallback=False, response_schema=None, max_retries=3):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if reply is FAILURE:
            raise GenerationError("RateLimitError: slow down")
        return LLMCall(text=reply if isinstance(reply, str) else json.dumps(reply))


def question(n):
    return {"answerable": True, "question": f"question {n}?", "answer": f"answer {n}."}


def run_generation(draft, groups, replies, count=10, difficulties=("easy", "hard"), sleeps=None):
    client = FakeClient(replies)
    sleeps = [] if sleeps is None else sleeps
    generate(draft, groups, client, count, difficulties, save=lambda d: None, progress=lambda _: None, sleep=sleeps.append)
    return client


def test_chunks_with_the_same_text_form_one_group_and_all_are_ground_truth():
    chunks = [
        source_chunk("a", doc_id="d1", text="Same  text\nhere", source="https://x/one", heading=("A",)),
        source_chunk("b", doc_id="d2", text="Same text here", source="https://x/two", heading=("B",)),
        source_chunk("c", doc_id="d2", text="different"),
    ]
    groups = {g.chunk_ids[0]: g for g in group_identical(chunks)}
    assert len(groups) == 2
    twins = groups["a"]
    assert twins.chunk_ids == ["a", "b"]  # whitespace differences do not make texts different
    assert twins.documents == ["https://x/one", "https://x/two"] and twins.headings == ["A", "B"]


def test_interleaving_spreads_the_sample_across_documents_and_follows_the_seed():
    groups = [group(f"big{i}", doc_id="big") for i in range(6)] + [group("s1", doc_id="small1"), group("s2", doc_id="small2")]
    first_three = {g.representative.doc_id for g in interleave_by_document(groups, seed=1)[:3]}
    assert first_three == {"big", "small1", "small2"}  # a long document does not crowd out the others
    assert [g.chunk_ids for g in interleave_by_document(groups, 1)] == [g.chunk_ids for g in interleave_by_document(groups, 1)]
    assert [g.chunk_ids for g in interleave_by_document(groups, 1)] != [g.chunk_ids for g in interleave_by_document(groups, 2)]
    assert len(interleave_by_document(groups, 1)) == len(groups)


def test_lexical_overlap_is_the_share_of_the_questions_words_found_in_the_source():
    source = "Rotate the signing keys every ninety days"
    assert lexical_overlap("rotate signing keys", source) == 1.0
    assert lexical_overlap("how often to renew credentials", source) == pytest.approx(0.0)
    assert lexical_overlap("rotate credentials", source) == pytest.approx(0.5)
    assert lexical_overlap("", source) == 0.0


def test_a_tier_that_is_not_harder_than_the_one_before_is_flagged():
    items = [DraftItem(query=f"q{i}", acceptable_documents=["d"], difficulty=tier, lexical_overlap=o)
             for i, (tier, o) in enumerate([("easy", 0.9), ("medium", 0.6), ("hard", 0.8)])]
    stats = overlap_by_difficulty(items)
    assert stats["medium"] == {"n": 1, "mean_overlap": 0.6}
    warnings = tier_warnings(stats)
    assert len(warnings) == 1 and "'hard'" in warnings[0] and "'medium'" in warnings[0]
    assert tier_warnings({"easy": {"n": 1, "mean_overlap": 0.9}, "hard": {"n": 1, "mean_overlap": 0.4}}) == []


def test_the_prompt_carries_the_passage_and_the_tiers_instruction():
    g = group("a", text="Keys rotate every 90 days.", heading=("Security", "Keys"))
    prompt = build_prompt(g, "hard")
    assert "Keys rotate every 90 days." in prompt and "Security > Keys" in prompt
    assert DIFFICULTY_INSTRUCTIONS["hard"] in prompt and DIFFICULTY_INSTRUCTIONS["easy"] not in prompt
    assert '{"answerable": true' in prompt  # the format example survived str.format


def test_replies_are_parsed_leniently_and_garbage_is_none():
    assert parse_generated('{"answerable": true, "question": "Q?", "answer": "A."}').question == "Q?"
    assert parse_generated('```json\n{"answerable": false}\n```').answerable is False
    assert parse_generated("I cannot do that") is None
    assert parse_generated("[1, 2]") is None
    assert parse_generated('{"answerable": "maybe"}') is None


def test_extract_json_object_rejects_what_is_not_an_object():
    assert extract_json_object('prefix {"a": 1} suffix') == {"a": 1}
    for bad in ("no json", "[1]", "{broken"):
        with pytest.raises(ValueError):
            extract_json_object(bad)


def test_generation_records_ground_truth_difficulty_overlap_and_category():
    twins = ChunkGroup((
        source_chunk("a", doc_id="d1", text="Run `make build` to compile.", code=True, source="https://x/one"),
        source_chunk("b", doc_id="d2", text="Run `make build` to compile.", code=True, source="https://x/two"),
    ))
    draft = Draft()
    run_generation(draft, [twins], [{"answerable": True, "question": "How do I compile?", "answer": "Run make build."}], count=1)

    item = draft.items[0]
    assert (item.source_chunk_ids, item.acceptable_documents) == (["a", "b"], ["https://x/one", "https://x/two"])
    assert (item.difficulty, item.category, item.origin, item.review_status) == ("easy", "code", SYNTHETIC, PENDING)
    assert item.lexical_overlap == pytest.approx(lexical_overlap("How do I compile?", item.source_text))
    assert item.source_text == "Run `make build` to compile."


def test_difficulties_cycle_and_the_count_is_respected():
    groups = [group(f"c{i}") for i in range(5)]
    draft = Draft()
    client = run_generation(draft, groups, [question(i) for i in range(5)], count=3, difficulties=("easy", "medium", "hard"))
    assert [i.difficulty for i in draft.items] == ["easy", "medium", "hard"]
    assert len(client.prompts) == 3  # no call beyond the requested count


def test_the_model_abstaining_or_repeating_a_question_adds_no_question_but_is_remembered():
    groups = [group("nav"), group("a"), group("dup")]
    draft = Draft()
    run_generation(draft, groups, [{"answerable": False}, question(1), question(1)], count=5)
    assert [i.query for i in draft.items] == ["question 1?"]
    assert draft.abstained == ["nav", "dup"]


def test_a_resumed_run_skips_chunks_already_handled_and_continues_the_tier_cycle():
    groups = [group(f"c{i}") for i in range(4)]
    draft = Draft()
    run_generation(draft, groups[:2], [question(0), {"answerable": False}], count=3)  # interrupted after c1
    resumed = run_generation(draft, groups, [question(1), question(2)], count=3, difficulties=("easy", "hard"))
    assert len(resumed.prompts) == 2  # c0 (a question) and c1 (abstained) are not asked again
    assert [i.source_chunk_ids for i in draft.items] == [["c0"], ["c2"], ["c3"]]
    assert [i.difficulty for i in draft.items] == ["easy", "hard", "easy"]


def test_failed_calls_are_retried_later_and_a_dead_provider_aborts_the_run():
    failure = FAILURE
    groups = [group(f"c{i}") for i in range(MAX_CONSECUTIVE_FAILURES + 1)]
    draft = Draft()
    with pytest.raises(GenerationAborted):
        run_generation(draft, groups, [failure] * MAX_CONSECUTIVE_FAILURES)
    assert draft.items == [] and draft.abstained == []  # nothing recorded, so a resume retries them

    recovering = Draft()
    run_generation(recovering, [group(f"m{i}") for i in range(8)], [failure] * (MAX_CONSECUTIVE_FAILURES - 1) + [question(1), failure, question(2)], count=2)
    assert [i.query for i in recovering.items] == ["question 1?", "question 2?"]  # the failure count resets after a success


def test_calls_are_paced_to_the_providers_rate():
    sleeps = []
    run_generation(Draft(), [group("a"), group("b")], [question(1), question(2)], sleeps=sleeps)
    assert len(sleeps) == 1 and 0.0 <= sleeps[0] <= 1.1  # one wait, between the two calls
