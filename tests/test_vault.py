from llm_redact.vault import InMemoryVault


def test_deterministic_same_value_same_placeholder(vault: InMemoryVault) -> None:
    a = vault.placeholder_for("EMAIL", "jane@example.com")
    b = vault.placeholder_for("EMAIL", "jane@example.com")
    assert a == b == "«EMAIL_001»"


def test_distinct_values_get_distinct_placeholders(vault: InMemoryVault) -> None:
    a = vault.placeholder_for("EMAIL", "jane@example.com")
    b = vault.placeholder_for("EMAIL", "john@example.com")
    assert a != b
    assert b == "«EMAIL_002»"


def test_same_value_different_type_distinct(vault: InMemoryVault) -> None:
    a = vault.placeholder_for("EMAIL", "x")
    b = vault.placeholder_for("SECRET", "x")
    assert a != b


def test_reverse_lookup(vault: InMemoryVault) -> None:
    token = vault.placeholder_for("EMAIL", "jane@example.com")
    assert vault.original_for(token) == "jane@example.com"


def test_unknown_placeholder_returns_none(vault: InMemoryVault) -> None:
    assert vault.original_for("«EMAIL_999»") is None
