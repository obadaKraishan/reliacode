import textwrap

import pytest

from reliacode._yaml import YamlError, load_yaml


def yaml_text(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def test_loads_data_and_records_lines_of_keys_and_items() -> None:
    loaded = load_yaml(
        yaml_text(
            """
            name: demo
            variables:
              - name: tone
                categories:
                  - code: positive
                  - code: negative
            """
        )
    )
    assert loaded.data["variables"][0]["categories"][1] == {"code": "negative"}
    assert loaded.lines[("name",)] == 1
    assert loaded.lines[("variables",)] == 2
    assert loaded.lines[("variables", 0)] == 3
    assert loaded.lines[("variables", 0, "categories", 1)] == 6
    assert loaded.lines[("variables", 0, "categories", 1, "code")] == 6


def test_line_for_falls_back_to_the_deepest_existing_part() -> None:
    loaded = load_yaml(
        yaml_text(
            """
            variables:
              - name: tone
                label: Tone
            """
        )
    )
    # "definition" is not in the document, so the variable's own line is used.
    assert loaded.line_for(("variables", 0, "definition")) == 2
    assert loaded.line_for(("nothing", "here")) == 1


def test_key_line_is_used_for_block_scalars() -> None:
    loaded = load_yaml(
        yaml_text(
            """
            name: demo
            description: |
              A long text
              over two lines.
            """
        )
    )
    assert loaded.data["description"] == "A long text\nover two lines.\n"
    assert loaded.lines[("description",)] == 2


def test_empty_text_gives_no_data() -> None:
    assert load_yaml("").data is None
    assert load_yaml("# only a comment\n").data is None


def test_duplicate_key_is_an_error_with_both_lines() -> None:
    text = yaml_text(
        """
        name: demo
        variables:
          - name: tone
            definition: first
            definition: second
        """
    )
    with pytest.raises(YamlError) as info:
        load_yaml(text)
    assert info.value.line == 5
    assert '"definition" appears twice' in info.value.message
    assert "lines 4 and 5" in info.value.message


def test_merge_keys_are_not_mistaken_for_duplicates() -> None:
    loaded = load_yaml(
        yaml_text(
            """
            base: &base {definition: shared}
            first:
              <<: *base
              code: a
            """
        )
    )
    assert loaded.data["first"] == {"definition": "shared", "code": "a"}


def test_colon_inside_text_gives_a_quoting_hint() -> None:
    with pytest.raises(YamlError) as info:
        load_yaml("definition: Conflict: disagreement between actors\n")
    assert info.value.line == 1
    assert "in quotes" in info.value.message


def test_tab_indentation_gives_a_spaces_hint() -> None:
    with pytest.raises(YamlError) as info:
        load_yaml("variables:\n\t- name: tone\n")
    assert info.value.line == 2
    assert "Indent with spaces" in info.value.message


def test_bad_indentation_gives_an_indentation_hint() -> None:
    text = yaml_text(
        """
        variables:
          - name: tone
           label: Tone
        """
    )
    with pytest.raises(YamlError) as info:
        load_yaml(text)
    assert info.value.line == 3
    assert "indentation" in info.value.message


def test_unclosed_quote_is_reported() -> None:
    with pytest.raises(YamlError) as info:
        load_yaml('name: "demo\n')
    assert "quote" in info.value.message


def test_several_documents_are_refused() -> None:
    with pytest.raises(YamlError) as info:
        load_yaml("name: one\n---\nname: two\n")
    assert "more than one YAML document" in info.value.message


def test_python_tags_are_refused() -> None:
    with pytest.raises(YamlError) as info:
        load_yaml("name: !!python/object/apply:os.system ['echo hi']\n")
    assert "not valid YAML" in info.value.message


def test_recursive_aliases_do_not_loop() -> None:
    loaded = load_yaml("items: &loop [1, *loop]\n")
    assert loaded.data["items"][0] == 1
    assert loaded.lines[("items", 1)] == 1
