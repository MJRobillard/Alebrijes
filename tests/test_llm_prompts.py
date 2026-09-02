from __future__ import annotations

from alebrijes.llm.prompts import ChatTurn, Mode, build_messages, build_system_prompt, load_guidelines


def test_load_guidelines_contains_no_spoiler_rule() -> None:
    text = load_guidelines()
    assert "spoiler" in text.lower()


def test_build_system_prompt_includes_guidelines_book_and_position() -> None:
    prompt = build_system_prompt(
        book_title="Fixture Book",
        character_voice="Narrator",
        position_summary="Chapter 2, partway through",
        rag_excerpts=["Colin met Fantine in the garden."],
        mode=Mode.QA,
        guidelines="GUIDELINES-MARKER",
    )

    assert "GUIDELINES-MARKER" in prompt
    assert "Fixture Book" in prompt
    assert "Narrator" in prompt
    assert "Chapter 2, partway through" in prompt
    assert "Colin met Fantine in the garden." in prompt


def test_build_system_prompt_handles_no_retrieved_excerpts() -> None:
    prompt = build_system_prompt(
        book_title="Fixture Book",
        character_voice="Narrator",
        position_summary="start",
        rag_excerpts=[],
        mode=Mode.QA,
        guidelines="g",
    )
    assert "no relevant excerpts retrieved" in prompt


def test_build_system_prompt_varies_by_mode() -> None:
    qa_prompt = build_system_prompt(
        book_title="B", character_voice="N", position_summary="p", rag_excerpts=[], mode=Mode.QA, guidelines="g"
    )
    cyoa_prompt = build_system_prompt(
        book_title="B", character_voice="N", position_summary="p", rag_excerpts=[], mode=Mode.CYOA, guidelines="g"
    )
    assert qa_prompt != cyoa_prompt


def test_build_messages_puts_system_first_then_history_then_new_message() -> None:
    history = [ChatTurn(role="user", content="who is Colin?"), ChatTurn(role="assistant", content="A traveler.")]

    messages = build_messages(
        mode=Mode.QA,
        book_title="Fixture Book",
        character_voice="Narrator",
        position_summary="Chapter 1",
        rag_excerpts=[],
        history=history,
        user_message="what did he say?",
        guidelines="g",
    )

    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "who is Colin?"}
    assert messages[2] == {"role": "assistant", "content": "A traveler."}
    assert messages[-1] == {"role": "user", "content": "what did he say?"}
