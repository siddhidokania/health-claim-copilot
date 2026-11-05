from app.agent import _extractive_answer, cited_ids, format_hits
from app.ingest.chunking import chunk_pages, looks_like_heading, paragraphs
from app.ingest.embedding import HashingEmbedder
from app.search import keyword_tsquery, rrf_merge

PAGES = [
    (1, "Medicare Part B\nPart B covers doctor visits and outpatient care.\n"
        "It also covers durable medical equipment.\n\nWhat Part A covers\n"
        "Part A covers inpatient hospital stays, skilled nursing facility care and hospice."),
    (2, "• Lab tests\n• Preventive services such as flu shots\n\n12\n"),
]


def hit(chunk_id, content="text", page=1, heading=None):
    return {"chunk_id": chunk_id, "document_id": 1, "title": "Medicare & You", "page": page,
            "heading": heading, "content": content, "source_url": None}


# ---------------------------------------------------------------- chunking

def test_heading_detection():
    assert looks_like_heading("What Part A covers")
    assert looks_like_heading("How do I sign up?")
    assert not looks_like_heading("Part B covers doctor visits and outpatient care.")
    assert not looks_like_heading("12")
    assert not looks_like_heading("• Lab tests")


def test_paragraphs_group_lines_bullets_and_drop_page_numbers():
    paras = paragraphs(PAGES)
    texts = [p.text for p in paras]
    assert texts[0] == "Medicare Part B" and paras[0].is_heading
    assert "Part B covers doctor visits and outpatient care. It also covers durable medical equipment." in texts
    assert "- Lab tests" in texts and "- Preventive services such as flu shots" in texts
    assert "12" not in texts


def test_chunks_keep_heading_and_page_and_respect_size():
    chunks = chunk_pages(PAGES, max_chars=120, overlap=30)
    assert chunks, "expected at least one chunk"
    assert all(len(c.content) <= 120 + 30 + 2 for c in chunks)
    part_a = next(c for c in chunks if "inpatient hospital" in c.content)
    assert part_a.heading == "What Part A covers" and part_a.page == 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_page_is_the_page_it_starts_on():
    pages = [(1, "Introduction\nMedicare is health insurance for people 65 or older and some younger people."),
             (2, "Part D Basics\nPart D helps cover the cost of prescription drugs, including many vaccines.")]
    chunks = chunk_pages(pages, max_chars=500, overlap=50)
    assert [(c.page, c.heading) for c in chunks] == [(1, "Introduction"), (2, "Part D Basics")]


def test_sentence_case_heading_needs_a_capitalised_next_line():
    paras = paragraphs([(1, "What hospice covers\nHospice covers pain relief and nursing care.\n\n"
                            "You can get hospice care if\nyour doctor certifies you are terminally ill.")])
    assert paras[0].is_heading and paras[0].text == "What hospice covers"
    assert not any(p.is_heading for p in paras[1:])


def test_overlap_carries_text_but_not_across_headings():
    long_text = " ".join(f"Sentence number {i} about Medicare coverage rules." for i in range(30))
    chunks = chunk_pages([(1, "Coverage Rules\n" + long_text)], max_chars=300, overlap=80)
    assert len(chunks) > 2
    # The start of each chunk after the first repeats the end of the previous one.
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.content.split("\n")[0] in prev.content
    # A new heading starts a clean chunk with no carried text.
    two = chunk_pages([(1, "First Topic\n" + long_text[:250] + "\n\nSecond Topic\nShort text about topic two "
                           "that is long enough to keep.")], max_chars=300, overlap=80)
    assert two[-1].heading == "Second Topic" and two[-1].content.startswith("Short text")


def test_no_duplicate_tail_chunk_at_end():
    text = "Coverage Basics\n" + " ".join(f"Line {i} explains a benefit in detail." for i in range(12))
    chunks = chunk_pages([(1, text)], max_chars=200, overlap=60)
    assert len({c.content for c in chunks}) == len(chunks)
    assert len(chunks[-1].content) > 60


# ---------------------------------------------------------------- retrieval helpers

def test_rrf_rewards_agreement_between_retrievers():
    fused = rrf_merge([[1, 2, 3], [3, 4, 1]], k=60)
    order = [doc for doc, _ in fused]
    assert order[0] == 1           # rank 1 and 3: found by both
    assert order[1] == 3           # rank 3 and 1
    assert set(order) == {1, 2, 3, 4}
    assert abs(fused[0][1] - (1 / 61 + 1 / 63)) < 1e-12


def test_rrf_weights_downgrade_a_noisy_retriever():
    # Item 9 is keyword-only at rank 1; item 5 is vector rank 2. With keyword weight 0.3, vector wins.
    fused = rrf_merge([[1, 5], [9, 1]], k=60, weights=[1.0, 0.3])
    assert [d for d, _ in fused] == [1, 5, 9]


def test_keyword_tsquery_ors_unique_words_and_strips_symbols():
    assert keyword_tsquery("What's the Part B   premium?? premium") == "part | premium"
    assert keyword_tsquery("Does Medicare cover hospice?") == "hospice"
    assert keyword_tsquery("!!!") == ""
    assert "'" not in keyword_tsquery("O'Neil; DROP TABLE chunks;--")


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(64)
    a, b = e.embed_query("hospice care"), e.embed_query("hospice care")
    assert a == b and abs(sum(x * x for x in a) - 1.0) < 1e-9
    assert e.embed_query("hospice care") != e.embed_query("diabetes supplies")


# ---------------------------------------------------------------- answer helpers

def test_cited_ids_keeps_order_and_ignores_unknown_ids():
    text = "Part B covers it [S12]. Also [S7] and again [S12]. Made up [S999]."
    assert cited_ids(text, {7, 12}) == [12, 7]


def test_format_hits_labels_sources():
    out = format_hits([hit(5, "Hospice covers pain relief.", page=3, heading="Hospice care")])
    assert out.startswith("[S5] Medicare & You, page 3 - Hospice care\n")
    assert format_hits([]) == "No matching passages."


def test_extractive_answer_cites_top_passages():
    text = _extractive_answer([hit(1, "a" * 500), hit(2, "b"), hit(3, "c"), hit(4, "d")])
    assert cited_ids(text, {1, 2, 3, 4}) == [1, 2, 3]
    assert "..." in text
    assert "couldn't find" in _extractive_answer([])




def test_index_pages_and_note_lines_are_not_headings():
    index = "\n".join(f"Topic number {i} {10 + i}" for i in range(12))
    assert paragraphs([(1, index)]) == []
    assert not looks_like_heading("Note: Contact your plan first")
    assert not looks_like_heading("Medicare Savings Programs 95 -96")
