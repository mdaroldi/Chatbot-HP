"""URL parameters: the client ID and the selected conversation, validated or replaced."""

import pytest

from hpdocs_ui.identifiers import trusted_identifier
from hpdocs_ui.url_state import CLIENT_PARAM, CONVERSATION_PARAM, UrlState, query_params, resolve

CID = "0b7e7a0c-2f7d-4f43-9d3f-6f2b8f1d2c3a"
CONV = "6f1c2d3e-4b5a-4c6d-8e7f-90a1b2c3d4e5"
NEW = "new-client-0001"


def fresh() -> str:
    return NEW


def test_the_parameter_names_are_short_and_fixed() -> None:
    assert (CLIENT_PARAM, CONVERSATION_PARAM) == ("cid", "conv")


def test_valid_parameters_are_kept() -> None:
    assert resolve({"cid": CID, "conv": CONV}, new_id=fresh) == UrlState(CID, CONV)


def test_a_conversation_id_is_written_in_canonical_form() -> None:
    assert resolve({"cid": CID, "conv": CONV.upper()}, new_id=fresh).conversation_id == CONV


def test_no_conversation_parameter_means_no_selection() -> None:
    assert resolve({"cid": CID}, new_id=fresh) == UrlState(CID, None)


@pytest.mark.parametrize("conv", ["", "not-a-uuid", CONV[:-1], CONV + "0", "<script>"])
def test_an_invalid_conversation_id_is_dropped_and_the_client_kept(conv: str) -> None:
    assert resolve({"cid": CID, "conv": conv}, new_id=fresh) == UrlState(CID, None)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"conv": CONV},
        {"cid": "", "conv": CONV},
        {"cid": "short7x", "conv": CONV},
        {"cid": "a" * 65, "conv": CONV},
        {"cid": "bad_char-0001", "conv": CONV},
        {"cid": "<b>x</b>-0001", "conv": CONV},
    ],
)
def test_a_missing_or_invalid_client_id_is_replaced_and_its_conversation_dropped(
    params: dict[str, str],
) -> None:
    assert resolve(params, new_id=fresh) == UrlState(NEW, None)


def test_the_default_new_client_id_is_a_trusted_identifier() -> None:
    client_id = resolve({}).client_id

    assert trusted_identifier(client_id) == client_id
    assert resolve({}).client_id != client_id


def test_query_params_carry_the_conversation_only_when_one_is_selected() -> None:
    assert query_params(UrlState(CID, CONV)) == {"cid": CID, "conv": CONV}
    assert query_params(UrlState(CID, None)) == {"cid": CID}
