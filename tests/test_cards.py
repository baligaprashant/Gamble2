from gamble2.vision.cards import Card, all_preflop_keys, canonical_preflop_key, parse_cards


def test_parse_card():
    assert Card.parse("Ah").code == "Ah"
    assert Card.parse("td").code == "Td"


def test_parse_cards():
    cards = parse_cards("Ah Kh")
    assert [c.code for c in cards] == ["Ah", "Kh"]


def test_canonical_keys():
    assert canonical_preflop_key(parse_cards("Ah Ad")) == "AA"
    assert canonical_preflop_key(parse_cards("Ah Kh")) == "AKs"
    assert canonical_preflop_key(parse_cards("Ah Kd")) == "AKo"
    assert canonical_preflop_key(parse_cards("Kh Ah")) == "AKs"


def test_169_keys():
    keys = all_preflop_keys()
    assert len(keys) == 169
    assert len(set(keys)) == 169
