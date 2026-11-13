"""The product-name rule: which manual a question names explicitly, if exactly one."""

import pytest

from app.domain.retrieval import ProductName, named_product

ENVY = ProductName(doc_id="envy-6000-user-guide", name="ENVY", model_number="6000")
OMEN = ProductName(doc_id="omen-17-service-guide", name="OMEN", model_number="17")
PRODUCTS = (ENVY, OMEN)


@pytest.mark.parametrize(
    "question",
    [
        "How do I clean the ENVY printhead?",
        "How do I clean the Envy printhead?",
        "Does the eNvY support duplex?",
        "What is the OMEN's battery capacity?",
        "Is the fan on the (Omen) loud?",
        "OMEN, bottom cover: which screws?",
        "ENVY-specific settings",
    ],
)
def test_a_capitalized_product_name_selects_its_manual(question: str) -> None:
    expected = OMEN.doc_id if "omen" in question.lower() else ENVY.doc_id

    assert named_product(question, PRODUCTS) == expected


@pytest.mark.parametrize(
    ("question", "doc_id"),
    [
        ("my envy 6000 jams", ENVY.doc_id),
        ("envy6000 wifi setup", ENVY.doc_id),
        ("the envy-6000 display", ENVY.doc_id),
        ("omen 17 keyboard removal", OMEN.doc_id),
        ("omen 17.3 inch display", OMEN.doc_id),
        ("OMEN 17 fan", OMEN.doc_id),
    ],
)
def test_a_name_in_any_case_followed_by_its_model_number_selects_its_manual(
    question: str, doc_id: str
) -> None:
    assert named_product(question, PRODUCTS) == doc_id


@pytest.mark.parametrize(
    "question",
    [
        "I envy people whose printers work",
        "a bad omen for the print queue",
        "envy",
        "omen 170 battery",  # another model number
        "envy 60001",
        "Omens and Envying are not product names",
        "PRENVY OMENS",
        "How do I replace the printer cartridge?",  # category words never fire
        "Which laptop screws hold the cover?",
        "",
    ],
)
def test_no_explicit_product_name_selects_nothing(question: str) -> None:
    assert named_product(question, PRODUCTS) is None


def test_both_products_named_selects_nothing() -> None:
    assert named_product("Can the OMEN print to the ENVY?", PRODUCTS) is None


def test_the_same_product_named_twice_still_selects_it() -> None:
    assert named_product("OMEN fan or omen 17 cooling?", PRODUCTS) == OMEN.doc_id


def test_a_product_without_a_model_number_matches_by_name_only() -> None:
    plain = ProductName(doc_id="demo-guide", name="Demo", model_number=None)

    assert named_product("the Demo setup", (plain,)) == "demo-guide"
    assert named_product("demo 100 setup", (plain,)) is None


def test_no_products_means_no_filter() -> None:
    assert named_product("OMEN fan", ()) is None


@pytest.mark.parametrize(
    ("name", "model_number"),
    [("", "6000"), ("  ", None), ("ENVY", ""), ("EN VY", None), ("ENVY", "60 00")],
)
def test_product_names_and_model_numbers_are_single_words(
    name: str, model_number: str | None
) -> None:
    with pytest.raises(ValueError, match="single word"):
        ProductName(doc_id="d", name=name, model_number=model_number)
