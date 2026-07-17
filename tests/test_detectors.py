import re

from llm_redact.detection.engine import (
    Allowlist,
    CustomRule,
    DetectionConfig,
    build_allowlist,
    build_detectors,
    detect_all,
)

NO_ALLOW = Allowlist(exact=frozenset(), patterns=())


def _types(text: str, config: DetectionConfig | None = None) -> list[str]:
    from llm_redact.redactor import _resolve_overlaps

    config = config or DetectionConfig()
    detections = detect_all(build_detectors(config), text, NO_ALLOW)
    return [d.detector_type for d in _resolve_overlaps(detections)]


def test_email_detected() -> None:
    assert _types("contact jane.doe+dev@corp.example.org today") == ["EMAIL"]


def test_ipv4_valid_only() -> None:
    assert _types("host 192.168.1.7 up") == ["IPV4"]
    assert _types("version 999.1.2.3") == []


def test_credit_card_luhn() -> None:
    assert _types("card 4111 1111 1111 1111 ok") == ["CREDIT_CARD"]
    # Same shape, fails Luhn.
    assert _types("card 4111 1111 1111 1112 ok") == []
    # All zeros satisfies Luhn arithmetic but is a placeholder, not a PAN.
    assert _types("card 0000 0000 0000 0000 ok") == []


def test_aws_access_key() -> None:
    assert _types("key AKIAIOSFODNN7EXAMPLE end") == ["AWS_KEY"]


def test_github_token() -> None:
    assert _types("tok ghp_" + "a1B2" * 9 + " end") == ["GITHUB_TOKEN"]


def test_anthropic_beats_openai_prefix() -> None:
    assert _types("sk-ant-api03-abcdefghijklmnopqrstuv") == ["ANTHROPIC_KEY"]


def test_openai_key() -> None:
    assert _types("sk-abcdefghijklmnopqrstuvwx") == ["OPENAI_KEY"]


def test_slack_token() -> None:
    assert _types("xoxb-1234567890-abcdefghij") == ["SLACK_TOKEN"]


def test_private_key_block_wins_whole() -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA7bq8V3xQ9c\nsk-abcdefghijklmnopqrstuvwx\n"
        "-----END RSA PRIVATE KEY-----"
    )
    detections = detect_all(build_detectors(DetectionConfig()), pem, NO_ALLOW)
    from llm_redact.redactor import _resolve_overlaps

    chosen = _resolve_overlaps(detections)
    assert [d.detector_type for d in chosen] == ["PRIVATE_KEY"]
    assert chosen[0].value == pem


def test_pgp_private_key_block_detected() -> None:
    # PGP/GPG armor puts " BLOCK" between "PRIVATE KEY" and the dashes, so the
    # plain PEM grammar missed it — a full private key leaked verbatim.
    pgp = (
        "-----BEGIN PGP PRIVATE KEY BLOCK-----\n"
        "Version: GnuPG v2\n\n"
        "lQOYBGABCDEFAQ\nsk-abcdefghijklmnopqrstuvwx\n"
        "-----END PGP PRIVATE KEY BLOCK-----"
    )
    from llm_redact.redactor import _resolve_overlaps

    chosen = _resolve_overlaps(detect_all(build_detectors(DetectionConfig()), pgp, NO_ALLOW))
    assert [d.detector_type for d in chosen] == ["PRIVATE_KEY"]
    assert chosen[0].value == pgp


def test_pgp_public_key_block_not_matched() -> None:
    # A PUBLIC key block is not a secret; it must NOT be redacted.
    pub = "-----BEGIN PGP PUBLIC KEY BLOCK-----\nmQENBGABCDEFAQ\n-----END PGP PUBLIC KEY BLOCK-----"
    assert "PRIVATE_KEY" not in _types(pub)


def test_eth_address_eip55_and_lowercase() -> None:
    # EIP-55 checksummed address (spec example) and its all-lowercase form.
    assert _types("wallet 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed here") == ["ETH_ADDRESS"]
    assert _types("wallet 0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed here") == ["ETH_ADDRESS"]
    # Mixed-case with a broken checksum is a typo, not an address.
    assert _types("wallet 0x5AAeb6053F3E94C9b9A09f33669435E7Ef1BeAed here") == []
    # The null/burn address is a placeholder, not a secret.
    assert _types("burn 0x0000000000000000000000000000000000000000 out") == []
    # 39 or 41 hex is not an address.
    assert _types("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beae here") == []


def test_btc_base58_and_bech32() -> None:
    assert _types("send to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa now") == ["BTC_ADDRESS"]
    assert _types("send to 3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy now") == ["BTC_ADDRESS"]
    assert _types("segwit bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4 ok") == ["BTC_ADDRESS"]
    assert _types("taproot bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0 ok") == [
        "BTC_ADDRESS"
    ]
    # A base58 string that fails the checksum does not fire.
    assert _types("addr 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNb now") == []
    # bech32 with a broken checksum does not fire.
    assert _types("addr bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5 now") == []


def test_korean_rrn_checksum_and_form() -> None:
    # 900101-1234568 has a valid mod-11 check digit (computed).
    assert _types("RRN 900101-1234568 registered") == ["KR_RRN"]
    assert _types("RRN 900101-1234569 registered") == []  # check digit wrong
    assert _types("RRN 901301-1234561 registered") == []  # month 13 invalid
    assert _types("RRN 900101-9234561 registered") == []  # gender digit 9
    assert _types("RRN 9001011234568 registered") == []  # bare run never fires


def test_singapore_nric_checksum() -> None:
    assert _types("NRIC S1234567D on file") == ["SG_NRIC"]  # canonical example
    assert _types("NRIC S1234567A on file") == []  # wrong check letter
    assert _types("NRIC T7654321B on file") == ["SG_NRIC"]  # T prefix adds 4
    assert _types("ref XS1234567D here") == []  # no boundary


def test_chinese_resident_id_checksum_and_gates() -> None:
    # GB 11643's own example pins the MOD 11-2 implementation.
    assert _types("id 11010519491231002X on file") == ["CN_RESIDENT_ID"]
    assert _types("id 11010519491231002x on file") == ["CN_RESIDENT_ID"]  # lowercase x
    assert _types("id 110105194912310021 on file") == []  # wrong check char
    assert _types("id 990105194912310025 on file") == []  # province 99 invalid
    # The grammar allows the SHAPE 0230 but the validator demands a real
    # calendar date — Feb 30 is rejected by datetime, not by regex.
    assert _types("id 110105194902300025 on file") == []
    assert _types("ref A11010519491231002X here") == []  # no boundary
    # An 18-digit value that is BOTH a valid CN id and Luhn-valid: the
    # ID-specific structure must win the exact-span tie with credit_card
    # (priority 90, the korean_rrn/french_nir rule).
    assert _types("value 110163199309147957 end") == ["CN_RESIDENT_ID"]


def test_round10_national_ids() -> None:
    # Japan My Number: 4-4-4 display form, ordinance mod-11 check digit.
    # Lead 0/1 stays outside aadhaar's lead-2-9 grammar.
    assert _types("number 1234 5678 9018 filed") == ["JP_MY_NUMBER"]
    assert _types("number 1234-5678-9018 filed") == ["JP_MY_NUMBER"]
    assert _types("number 1234 5678 9012 filed") == []  # wrong check digit
    assert _types("solid 123456789018 never fires") == []
    # A 4-4-4 SUBSPAN of a longer separated run must not fire: the first
    # three groups of this Luhn-FAILING card happen to pass the mod-11
    # check, and the adjacency lookarounds are what keep it out.
    assert _types("card 4111 1111 1111 1112 ok") == []

    # Thai Citizen ID: dashed 1-4-5-2-1 display form, folded mod-11 check.
    assert _types("id 1-1017-00230-70-8 attached") == ["TH_ID"]
    assert _types("id 1-1017-00230-70-9 attached") == []  # wrong check digit
    assert _types("solid 1101700230708 never fires") == []
    # Valid as BOTH a Thai ID and a Luhn number: the ID-specific grouping
    # wins the exact-span tie with credit_card (priority 90).
    assert _types("id 5-1819-09378-65-8 end") == ["TH_ID"]

    # Irish PPSN: mod-23 check letter; legacy W ignored, post-2013 A/H
    # weighted at 9x.
    assert _types("ppsn 1234567T on file") == ["IE_PPS"]
    assert _types("ppsn 1234567TW on file") == ["IE_PPS"]
    assert _types("ppsn 1234567FA on file") == ["IE_PPS"]
    assert _types("ppsn 1234567A on file") == []  # wrong check letter
    assert _types("ppsn 1234567FW on file") == []  # W contributes 0, not A's 9

    # Mexican CURP: state gate + REAL calendar date (century from the
    # homoclave: digit = 1900s, letter = 2000s) + mod-10 over the
    # N-tilde-bearing RENAPO charset.
    assert _types("curp GOMC900514HDFMRR05 ok") == ["MX_CURP"]
    assert _types("curp GOMC900514HDFMRR04 ok") == []  # wrong check digit
    assert _types("curp GOMC900230HDFMRR04 ok") == []  # Feb 30, correct check
    assert _types("curp XAXA000229HXXXXX15 ok") == []  # state XX invalid
    # Same date digits, opposite centuries: 1900 was NOT a leap year
    # (digit homoclave rejected), 2000 was (letter homoclave accepted).
    assert _types("curp XAXA000229HDFXXX14 no") == []
    assert _types("curp XAXA000229HDFXXXA6 ok") == ["MX_CURP"]


def test_round10_vendor_tokens() -> None:
    assert _types("TAVILY_API_KEY=tvly-dev-Abc123Def456Ghi789Jkl0 set") == ["TAVILY_KEY"]
    assert _types("key fc-0123456789abcdef0123456789abcdef used") == ["FIRECRAWL_KEY"]
    assert _types("key fc-0123456789abcdef0123456789abcde used") == []  # 31 hex: too short
    assert _types("fc-first-quarter results") == []  # prose after the short prefix
    nv = "nvapi-" + "Ab0_-" * 12
    assert _types(f"header {nv} sent") == ["NVIDIA_KEY"]
    assert _types("csk-" + "x1" * 20 + " cerebras") == ["CEREBRAS_KEY"]
    # Langfuse secret key beats the generic sk- rule on the shared span;
    # the pk-lf- public key is deliberately unmatched.
    lf = "sk-lf-01234567-89ab-cdef-0123-456789abcdef"
    assert _types(f"export LANGFUSE_SECRET_KEY={lf}") == ["LANGFUSE_KEY"]
    assert _types("pk-lf-01234567-89ab-cdef-0123-456789abcdef") == []
    assert _types("figd_" + "A1-b" * 10 + " figma") == ["FIGMA_PAT"]


def test_round9_vendor_tokens() -> None:
    assert _types("key NRAK-ABCDEFGHIJKLMNOPQRSTUVWXY01 end") == ["NEW_RELIC_KEY"]
    assert _types("tok glsa_" + "a" * 32 + "_0f1e2d3c end") == ["GRAFANA_TOKEN"]
    assert _types("key jina_" + "b" * 64 + " end") == ["JINA_KEY"]
    assert _types("bot 123456789:" + "C" * 35 + " up") == ["TELEGRAM_BOT_TOKEN"]
    # Near-misses: wrong prefix casing / short body do not fire.
    assert _types("nrak-ABCDEFGHIJKLMNOPQRSTUVWXY01 end") == []
    assert _types("bot 123:abc up") == []  # not the 8-10 digit : 35-char shape


def test_generic_secret_entropy_gate() -> None:
    assert _types('api_key = "hunter2hunter2hunter2"') == []  # low entropy
    assert _types('api_key = "9xK2mPqR7vT4wN8jL3hF6bXz"') == ["SECRET"]


def test_generic_secret_requires_assignment_context() -> None:
    # A bare high-entropy string with no keyword context is left alone.
    assert _types("9xK2mPqR7vT4wN8jL3hF6bXz") == []


def test_allowlist_exact_and_pattern() -> None:
    allow = build_allowlist(
        DetectionConfig(allowlist=("noreply@github.com",), allowlist_patterns=(r"^10\.",))
    )
    detectors = build_detectors(DetectionConfig())
    assert detect_all(detectors, "noreply@github.com", allow) == []
    assert detect_all(detectors, "10.0.0.5", allow) == []
    assert detect_all(detectors, "127.0.0.1", allow) == []  # default allowlist
    assert len(detect_all(detectors, "8.8.8.8", allow)) == 1


def test_canadian_sin_luhn_and_area() -> None:
    # 046 454 286 is the classic Luhn-valid example SIN.
    assert _types("SIN 046 454 286 on file") == ["CA_SIN"]
    assert _types("SIN 046-454-286 on file") == ["CA_SIN"]
    assert _types("SIN 046 454 287 on file") == []  # Luhn fails
    assert _types("SIN 123 456 789 on file") == []  # Luhn fails (doc classic)
    assert _types("SIN 846 454 286 on file") == []  # area 8: never assigned
    assert _types("SIN 046454286 on file") == []  # bare digits never fire
    assert _types("SIN 046 454-286 on file") == []  # mixed separators


def test_uk_nino_grammar() -> None:
    assert _types("NI number AB123456C given") == ["UK_NINO"]
    assert _types("NI number AB 12 34 56 C given") == ["UK_NINO"]
    assert _types("NI number BG123456C given") == []  # invalid prefix pair
    assert _types("NI number DA123456C given") == []  # D never first
    assert _types("NI number AO123456C given") == []  # O never second
    # HMRC reserves Q-prefixes for documentation examples (QQ 12 34 56 C);
    # the grammar excludes Q entirely, so the doc example never fires.
    assert _types("NI number QQ123456C given") == []
    assert _types("NI number AB123456E given") == []  # suffix must be A-D
    assert _types("NI number ab123456c given") == []  # uppercase only
    assert _types("ref XAB123456C given") == []  # no boundary


def test_aadhaar_verhoeff_and_grouping() -> None:
    # Verhoeff-valid 4-4-4 vectors (check digit computed).
    assert _types("Aadhaar 2345 6789 0124 on file") == ["AADHAAR"]
    assert _types("Aadhaar 6268-4656-3217 on file") == ["AADHAAR"]
    assert _types("Aadhaar 2345 6789 0120 on file") == []  # Verhoeff fails
    assert _types("Aadhaar 1345 6789 0124 on file") == []  # 0/1 never lead
    assert _types("Aadhaar 234567890124 on file") == []  # bare digits never fire
    assert _types("Aadhaar 2345 6789-0124 on file") == []  # mixed separators


def test_australian_tfn_checksum() -> None:
    # ATO weighted mod-11 vectors (leading 8 keeps them out of _sin_ok's
    # reach — canadian_sin wins exact-span ties by registration order).
    assert _types("TFN 861 318 607 quoted") == ["AU_TFN"]
    assert _types("TFN 852-601-819 quoted") == ["AU_TFN"]
    assert _types("TFN 861 318 608 quoted") == []  # checksum fails
    assert _types("TFN 861318607 quoted") == []  # bare digits never fire
    # A value passing BOTH checksums is claimed by the earlier rule — the
    # string itself is ambiguous, and one redaction is all that matters.
    assert _types("id 046 454 286 here") == ["CA_SIN"]


def test_spanish_dni_control_letter() -> None:
    assert _types("DNI 12345678Z presentado") == ["ES_DNI"]
    assert _types("DNI 12345678-Z presentado") == ["ES_DNI"]
    assert _types("NIE X7654321J presentado") == ["ES_DNI"]
    assert _types("DNI 12345678A presentado") == []  # wrong letter
    assert _types("NIE X7654321Z presentado") == []  # wrong letter
    assert _types("ref A12345678Z here") == []  # no boundary
    assert _types("DNI 12345678 Z presentado") == []  # spaced letter never fires


def test_french_nir_key() -> None:
    assert _types("NIR 2 08 02 75 968 277 14 au dossier") == ["FR_NIR"]
    assert _types("NIR 1 85 03 75 116 384 32 au dossier") == ["FR_NIR"]
    assert _types("NIR 2 69 09 2B 488 407 45 au dossier") == ["FR_NIR"]  # Corsica
    assert _types("NIR 1 85 03 75 116 384 69 au dossier") == []  # key fails
    assert _types("NIR 185037511638432 au dossier") == []  # bare digits never fire
    assert _types("NIR 3 85 03 75 116 384 32 au dossier") == []  # sex digit 1|2


def test_french_nir_beats_credit_card_on_ties() -> None:
    # 1/10 of NIRs are also Luhn-valid; the loose 13-19-digit card grammar
    # covers the whole spaced NIR, so the tie must go to the NIR-specific
    # grouping (priority 90). These vectors pass BOTH checksums.
    assert _types("NIR 1 76 07 82 988 464 07 au dossier") == ["FR_NIR"]
    assert _types("NIR 2 32 07 91 844 927 04 au dossier") == ["FR_NIR"]


def test_swiss_ahv_ean13() -> None:
    # The official documentation example — pins the EAN-13 transcriptions.
    assert _types("AHV 756.9217.0769.85 registriert") == ["CH_AHV"]
    assert _types("AHV 756.9217.0769.86 registriert") == []  # check fails
    assert _types("AHV 7569217076985 registriert") == []  # bare digits never fire
    assert _types("id 757.9217.0769.85 x") == []  # wrong country prefix


def test_swedish_personnummer_luhn_and_ranges() -> None:
    # Hand-computed Luhn-valid value (fake person, structurally valid).
    assert _types("pnr 850709-9870 anges") == ["SE_PNR"]
    assert _types("pnr 850709-9871 anges") == []  # Luhn fails
    # Coordination numbers add 60 to the day and are real identifiers
    # (hand-computed Luhn check digit 7 for this one).
    assert _types("pnr 850769-9877 anges") == ["SE_PNR"]
    assert _types("pnr 851309-9870 anges") == []  # month 13 invalid
    assert _types("pnr 8507099870 anges") == []  # bare digits never fire


def test_brazilian_cpf_check_digits() -> None:
    # 111.444.777-35 is the standard documentation example (both mod-11
    # digits check out by hand).
    assert _types("CPF 111.444.777-35 do cliente") == ["BR_CPF"]
    assert _types("CPF 111.444.777-36 do cliente") == []  # check digit fails
    # All-same-digit numbers pass the arithmetic but are never issued.
    assert _types("CPF 111.111.111-11 do cliente") == []
    assert _types("CPF 11144477735 do cliente") == []  # bare digits never fire


def test_italian_codice_fiscale_check_letter() -> None:
    # The canonical documentation example (Mario Rossi) — pins the rule's
    # odd-position table against the corpus's independent transcription.
    assert _types("CF RSSMRA85T10A562S registrato") == ["IT_CF"]
    # Same person, female day encoding (+40).
    assert _types("CF RSSMRA85T50A562W registrata") == ["IT_CF"]
    assert _types("CF RSSMRA85T10A562T registrato") == []  # check letter fails
    # Day 32 is structurally invalid even with a correct check letter (Y).
    assert _types("CF RSSMRA85T32A562Y registrato") == []
    # Month letter outside ABCDEHLMPRST never matches the grammar.
    assert _types("CF RSSMRA85F10A562S registrato") == []


def test_german_steuer_id_structure_and_check() -> None:
    assert _types("Steuer-ID 82 731 589 462 gemeldet") == ["DE_STEUER_ID"]
    assert _types("Steuer-ID 45 370 641 284 gemeldet") == ["DE_STEUER_ID"]
    assert _types("Steuer-ID 82 731 589 463 gemeldet") == []  # check fails
    assert _types("Steuer-ID 82731589462 gemeldet") == []  # bare digits never fire
    # First ten digits all distinct violates the repeat rule even when the
    # ISO 7064 arithmetic works out (3 is the correct check for 1234567890).
    assert _types("Steuer-ID 12 345 678 903 gemeldet") == []
    allow = build_allowlist(DetectionConfig(allowlist_by_type=(("EMAIL", ("ceo@corp.example",)),)))
    detectors = build_detectors(DetectionConfig())
    # Allowed when matched as EMAIL...
    assert detect_all(detectors, "mail ceo@corp.example now", allow) == []
    # ...but other EMAIL values still fire, and other types are unaffected.
    assert len(detect_all(detectors, "mail cfo@corp.example now", allow)) == 1
    assert len(detect_all(detectors, "host 8.8.8.8 up", allow)) == 1


def test_belgian_nn_mod97_and_century_prefix() -> None:
    # Dotted display form; two check digits = 97 - (body mod 97).
    assert _types("NN 93.05.18-223.61 geregistreerd") == ["BE_NN"]
    assert _types("NN 93.05.18-223.62 geregistreerd") == []  # check off by one
    # Births from 2000 on prepend '2' to the body before the modulus.
    assert _types("NN 01.01.01-001.26 geregistreerd") == ["BE_NN"]
    assert _types("NN 93.13.18-223.61 geregistreerd") == []  # month 13 invalid
    assert _types("NN 93051822361 geregistreerd") == []  # bare digits never fire


def test_finnish_hetu_mod31_check_character() -> None:
    # DDMMYY + century sign + 3-digit serial + mod-31 check character.
    assert _types("HETU 131052-308T tunnus") == ["FI_HETU"]
    assert _types("HETU 131052-308U tunnus") == []  # wrong check character
    # 2000s birth uses an A-F century sign (here 'A').
    assert _types("HETU 131002A308W tunnus") == ["FI_HETU"]
    assert _types("HETU 131352-308T tunnus") == []  # month 13 invalid


def test_nhs_number_mod11_spaced_form() -> None:
    # 3-3-4 spaced display form, mod-11 check. 943 476 5919 is the standard
    # published example.
    assert _types("NHS 943 476 5919 registered") == ["NHS_NUMBER"]
    assert _types("NHS 943 476 5918 registered") == []  # check digit fails
    assert _types("NHS 9434765919 registered") == []  # bare digits never fire
    # Only the space-separated form is NHS; the hyphenated form is a phone shape.
    assert "NHS_NUMBER" not in _types("NHS 943-476-5919 registered")


def test_norwegian_fnr_double_mod11() -> None:
    # DDMMYY-NNNNN, two mod-11 control digits; spaced or hyphenated.
    assert _types("FNR 150385 50060 oppgitt") == ["NO_FNR"]
    assert _types("FNR 150385-50060 oppgitt") == ["NO_FNR"]
    assert _types("FNR 150385 50061 oppgitt") == []  # second check digit fails
    assert _types("FNR 15038550060 oppgitt") == []  # bare digits never fire


def test_deny_bypasses_allowlist_by_type() -> None:
    from llm_redact.detection.deny import DenyEntry

    config = DetectionConfig(
        allowlist_by_type=(("DENY", ("ceo@corp.example",)),),
        deny_strings=(DenyEntry(value="ceo@corp.example"),),
    )
    detections = detect_all(
        build_detectors(config), "mail ceo@corp.example", build_allowlist(config)
    )
    # The deny match survives every allowlist — global and per-type.
    assert any(d.tier == 0 for d in detections)


def test_custom_rule() -> None:
    config = DetectionConfig(
        custom_rules=(CustomRule(name="jira", detector_type="TICKET", pattern=r"PROJ-\d{4,}"),)
    )
    assert _types("see PROJ-12345", config) == ["TICKET"]


def test_unknown_enabled_rule_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown detection rule"):
        build_detectors(DetectionConfig(enabled=("nope",)))


def test_detections_sorted_for_greedy_sweep() -> None:
    text = "sk-ant-api03-abcdefghijklmnopqrstuv"
    detections = detect_all(build_detectors(DetectionConfig()), text, NO_ALLOW)
    # Both sk- and sk-ant- rules match at 0; longer/higher-priority first.
    assert detections[0].detector_type == "ANTHROPIC_KEY"
    assert re.match(r"sk-ant-", detections[0].value)


# ---- Phase 3 vendor rules ----


def test_google_api_key() -> None:
    assert _types("key AIzaSyA1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tUvW") == ["GOOGLE_API_KEY"]
    assert _types("AIza-too-short") == []


def test_gcp_private_key_id() -> None:
    text = '{"private_key_id": "0123456789abcdef0123456789abcdef01234567"}'
    detections = detect_all(build_detectors(DetectionConfig()), text, NO_ALLOW)
    gcp = [d for d in detections if d.detector_type == "GCP_KEY_ID"]
    assert len(gcp) == 1
    assert gcp[0].value == "0123456789abcdef0123456789abcdef01234567"


def test_gcp_service_account_pem_with_escapes_still_caught() -> None:
    # GCP service-account JSON carries the PEM with literal \n escapes inside
    # the JSON string; the DOTALL private_key rule must still match.
    text = '{"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvA\\n-----END PRIVATE KEY-----\\n"}'
    assert "PRIVATE_KEY" in _types(text)


def test_azure_storage_key() -> None:
    import base64

    key = base64.b64encode(bytes(range(48))).decode()  # 64 chars, valid base64
    assert _types(f"DefaultEndpointsProtocol=https;AccountKey={key};EndpointSuffix=x") == [
        "AZURE_STORAGE_KEY"
    ]
    # Low-entropy / bad-length values rejected.
    assert _types("AccountKey=" + "a" * 64) == []


def test_jwt_with_valid_header() -> None:
    import base64
    import json as _json

    header = base64.urlsafe_b64encode(_json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(_json.dumps({"sub": "1234567890"}).encode()).decode()
    payload = payload.rstrip("=")
    token = f"{header}.{payload}.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert _types(f"Authorization: Bearer {token}") == ["JWT"]


def test_jwt_prose_decoy_rejected() -> None:
    # eyJ-shaped but the first segment is not valid base64url JSON.
    assert _types("eyJnotarealheaderzz.zzzzzzzzzzzzzz.zzzzzz") == []


def test_stripe_live_only() -> None:
    assert _types("sk_live_" + "a1B2c3D4e5F6g7H8") == ["STRIPE_KEY"]
    assert _types("rk_live_" + "a1B2c3D4e5F6g7H8") == ["STRIPE_KEY"]
    assert _types("sk_test_" + "a1B2c3D4e5F6g7H8") == []


def test_sendgrid_key() -> None:
    token = "SG." + "a" * 11 + "B" * 11 + "." + "c" * 21 + "D" * 22
    assert _types(f"key: {token}") == ["SENDGRID_KEY"]


def test_twilio_id_and_git_sha_decoy() -> None:
    assert _types("sid AC0123456789abcdef0123456789abcdef") == ["TWILIO_ID"]
    # A 40-char git SHA contains no AC/SK prefix at a word boundary.
    assert _types("commit 4ac3512058478d08beb8633ab344dece708267d3") == []


def test_npm_and_pypi_tokens() -> None:
    assert _types("npm_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8") == ["NPM_TOKEN"]
    pypi = "pypi-AgEIcHlwaS5vcmc" + "C2dE3fG4hI5jK6lM7nO8pQ9rS0tUvW1xY2zA3bC4dE5fG6hJ7kL8"
    assert _types(pypi) == ["PYPI_TOKEN"]


def test_github_fine_grained_pat() -> None:
    assert _types("github_pat_" + "a1B2c3D4e5F6g7H8i9J0k1" + "_extra") == ["GITHUB_TOKEN"]


def test_huggingface_entropy_gate() -> None:
    assert _types("hf_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6") == ["HF_TOKEN"]
    assert _types("hf_" + "a" * 30) == []  # low entropy


def test_openrouter_beats_generic_sk() -> None:
    assert _types("sk-or-v1-" + "0123456789abcdef" * 3) == ["OPENROUTER_KEY"]


def test_groq_key() -> None:
    assert _types("gsk_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8s9T0") == ["GROQ_KEY"]


def test_jwt_beats_generic_secret_on_tie() -> None:
    import base64
    import json as _json

    header = base64.urlsafe_b64encode(_json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
    token = f"{header}.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhb"
    # Keyword context makes generic_secret match the same span; JWT wins.
    types = _types(f"token = {token}")
    assert "JWT" in types
    assert "SECRET" not in types


def test_phone_formats_with_separators() -> None:
    assert _types("call +1 415 555 0100 today") == ["PHONE"]
    assert _types("or +442071838750 works") == ["PHONE"]
    assert _types("US style (212) 555-0100 too") == ["PHONE"]
    assert _types("hyphens 212-555-0100 fine") == ["PHONE"]
    assert _types("dots 212.555.0100 fine") == ["PHONE"]


def test_phone_bare_digits_never_fire() -> None:
    assert _types("invoice 4155550100 paid") == []
    assert _types("order 2125550100123 shipped") == []
    assert _types("call 212-555-01 back") == []  # too short


def test_ssn_hyphenated_with_valid_ranges() -> None:
    assert _types("SSN 078-05-1120 on file") == ["SSN"]
    assert _types("SSN 000-05-1120 on file") == []  # invalid area
    assert _types("SSN 666-05-1120 on file") == []
    assert _types("SSN 912-05-1120 on file") == []  # 9xx area
    assert _types("SSN 078-00-1120 on file") == []  # invalid group
    assert _types("SSN 078-05-0000 on file") == []  # invalid serial
    assert _types("ticket 078-05-11207 open") == []  # longer run


def test_iban_mod97_checksum() -> None:
    assert _types("pay DE89370400440532013000 now") == ["IBAN"]  # textbook valid
    assert _types("pay GB29NWBK60161331926819 now") == ["IBAN"]
    assert _types("pay DE89370400440532013001 now") == []  # checksum off by one
    assert _types("pay DE8937040044053201300 now") == []  # wrong length for DE


def test_gitlab_pat() -> None:
    assert _types("tok glpat-" + "a1B2" * 6 + " end") == ["GITLAB_TOKEN"]
    assert _types("tok glpat-short end") == []


def test_databricks_token() -> None:
    assert _types("dapi" + "0123456789abcdef" * 2 + " end") == ["DATABRICKS_TOKEN"]
    assert _types("dapi0123abc end") == []
    assert _types("dapi" + "ABCDEF0123456789" * 2 + " end") == []  # lowercase hex only


def test_bitbucket_app_password() -> None:
    assert _types("pw ATBB" + "a1B2" * 6 + " end") == ["BITBUCKET_TOKEN"]
    assert _types("pw ATBBshort1 end") == []


def test_atlassian_api_token() -> None:
    assert _types("tok ATATT" + "a1B2" * 6 + "= end") == ["ATLASSIAN_TOKEN"]
    assert _types("tok ATATTx end") == []


def test_round4_vendor_tokens_fire() -> None:
    hex64 = "0123456789abcdef" * 4
    hex32 = "0123456789abcdef" * 2
    alnum40 = "a1B2c3D4e5" * 4
    vectors = {
        "TAILSCALE_KEY": "tskey-api-kMhk9E7iXm1-" + alnum40[:24],
        "DO_TOKEN": "dop_v1_" + hex64,
        "NOTION_TOKEN": "secret_" + alnum40 + "xyZ",  # exactly 43
        "LINEAR_KEY": "lin_api_" + alnum40,
        "SUPABASE_KEY": "sbp_" + hex64[:40],
        "PLANETSCALE_TOKEN": "pscale_pw_" + alnum40[:20] + "._-" + alnum40[:12],
        "DOPPLER_TOKEN": "dp.pt." + alnum40 + "xyZ",
        "POSTMAN_KEY": "PMAK-" + hex64[:24] + "-" + hex64[:34],
        "AIRTABLE_PAT": "pat" + alnum40[:14] + "." + hex64,
        "SHOPIFY_TOKEN": "shpat_" + hex32,
    }
    for expected, token in vectors.items():
        assert _types(f"value {token} here") == [expected], expected


def test_round4_vendor_tokens_near_misses_stay_quiet() -> None:
    hex64 = "0123456789abcdef" * 4
    probes = [
        "tskey-api-k12",  # tail too short
        "dop_v2_" + hex64,  # no such prefix family
        "dop_v1_" + hex64[:63],  # 63 hex, one short
        "secret_prod_backend_v2",  # identifier, not 43 alnum
        "secret_" + "a1B2c3D4e5" * 4,  # 40 alnum, wrong length
        "lin_api_short",
        "sb_publishable_" + "a1B2c3D4e5" * 3,  # deliberately not a secret
        "dp.pt.example",
        "PMAK-" + hex64[:24] + "-0123",  # second half too short
        "pat" + "a1B2c3D4e5"[:9] + "." + hex64,  # id half too short
        "dispatch.pattern.path",  # "pat" inside ordinary words
        "shpat_" + hex64[:16],  # 16 hex, half length
    ]
    for probe in probes:
        assert _types(f"value {probe} here") == [], probe


def test_round7_vendor_tokens_fire() -> None:
    alnum80 = "a1B2c3D4e5" * 8
    vectors = {
        "GITLAB_TOKEN": "glrt-" + alnum80[:22],  # runner token, glpat sibling
        "GOOGLE_OAUTH_SECRET": "GOCSPX-" + alnum80[:28],
        "SENTRY_TOKEN": "sntrys_" + alnum80[:44],
        "XAI_KEY": "xai-" + alnum80,
        "PERPLEXITY_KEY": "pplx-" + alnum80[:48],
    }
    for expected, token in vectors.items():
        assert _types(f"value {token} here") == [expected], expected
    # Every GitLab routable-token prefix in the family fires.
    for prefix in ("glcbt-", "gldt-", "glptt-", "glagent-", "glimt-", "glsoat-"):
        assert _types(f"tok {prefix}{alnum80[:22]} end") == ["GITLAB_TOKEN"], prefix


def test_round7_vendor_tokens_near_misses_stay_quiet() -> None:
    alnum80 = "a1B2c3D4e5" * 8
    probes = [
        "glxx-" + alnum80[:22],  # not a real GitLab prefix family
        "glrt-short",  # tail under 20 chars
        "GOCSPX-tooshort",  # tail under 24 chars
        "sntryx_" + alnum80[:44],  # only sntrys_/sntryu_ exist
        "xai-" + alnum80[:39],  # 39 chars, one short of 40
        "pplx-" + alnum80[:39],  # 39 chars, one short
    ]
    for probe in probes:
        assert _types(f"value {probe} here") == [], probe


def test_asia_temporary_access_key() -> None:
    # STS temporary keys use the ASIA prefix; same AWS_KEY type as AKIA.
    assert _types("key ASIAIOSFODNN7EXAMPLE end") == ["AWS_KEY"]
    assert _types("key AKIAIOSFODNN7EXAMPLE end") == ["AWS_KEY"]


def test_round8_vendor_tokens_fire() -> None:
    alnum40 = "a1B2c3D4e5" * 4
    vectors = {
        "VAULT_TOKEN": "hvs." + alnum40[:26],
        "LANGSMITH_KEY": "lsv2_pt_" + alnum40[:36],
        "REPLICATE_TOKEN": "r8_" + alnum40[:37],
        "PINECONE_KEY": "pcsk_" + alnum40[:30],
    }
    for expected, token in vectors.items():
        assert _types(f"value {token} here") == [expected], expected
    assert _types("tok hvb." + alnum40[:26] + " end") == ["VAULT_TOKEN"]  # batch variant
    assert _types("tok lsv2_sk_" + alnum40[:36] + " end") == ["LANGSMITH_KEY"]


def test_round8_vendor_tokens_near_misses_stay_quiet() -> None:
    alnum40 = "a1B2c3D4e5" * 4
    probes = [
        "hvx." + alnum40[:26],  # only hvs./hvb.
        "lsv2_xx_" + alnum40[:36],  # only pt/sk
        "r8_" + alnum40[:20],  # tail too short (needs 37)
        "pcsk_short",  # tail too short
    ]
    for probe in probes:
        assert _types(f"value {probe} here") == [], probe


def test_ipv6_detected() -> None:
    assert _types("host 2001:db8:85a3::8a2e:370:7334 up") == ["IPV6"]
    assert _types("gw 2001:db8:1:2:3:4:5:6 up") == ["IPV6"]


def test_ipv6_decoys_rejected() -> None:
    assert _types("x[::2] slice") == []  # letterless, <4 groups
    assert _types("serial 04:9f:86:d0:81:88:4c:7d ok") == []  # 8 hex pairs
    assert _types("at 08:00:01 today") == []  # not parseable as IPv6
    assert _types("mac 00:1b:44:11:3a:b7 seen") == []  # 6 groups, no ::
    assert _types("ratio 1:2:3") == []  # letterless, 3 groups


def test_ipv6_loopback_in_default_allowlist() -> None:
    allow = build_allowlist(DetectionConfig())
    assert detect_all(build_detectors(DetectionConfig()), "bind to ::1 now", allow) == []


def test_url_credentials_password_only() -> None:
    from llm_redact.redactor import _resolve_overlaps

    text = "dsn postgres://svc:Xk29$QmPl40Vt@db3.internal:5432/app end"
    detections = _resolve_overlaps(detect_all(build_detectors(DetectionConfig()), text, NO_ALLOW))
    assert [(d.detector_type, d.value) for d in detections] == [("URL_PASSWORD", "Xk29$QmPl40Vt")]


def test_url_credentials_alnum_password_engulfed_by_email_is_still_redacted() -> None:
    # Deliberate: an all-alnum password + @host forms an email shape and the
    # longer EMAIL match wins the overlap — the secret is still redacted,
    # just typed EMAIL. Pinned so a future sort change is a conscious one.
    text = "dsn postgres://svc:Xk29QmPl40Vt@db3.internal:5432/app end"
    types = _types(text)
    assert types == ["EMAIL"]


def test_url_credentials_negatives() -> None:
    assert _types("see https://api.example.com:8443/v2") == []  # port, not a password
    # user@dotted-host (no password) is not URL_PASSWORD; the email grammar
    # does claim it — accepted over-redaction, pinned in the fp manifest too.
    assert _types("pull ssh://git@build.internal/repo.git") == ["EMAIL"]
    assert _types("plain http://user@dothost/path") == []  # user only, no TLD shape


def test_example_config_enabled_list_covers_every_builtin_rule() -> None:
    # config.example.toml's enabled list is COMMENTED OUT (3.4.0): a live
    # list froze the rule set for anyone copying the example — rules added
    # later silently never activated (chinese_resident_id fell through this
    # crack in 1.14.0). Pin both halves: no live list, and the commented
    # list still names every builtin rule so uncommenting it is complete.
    import re
    from pathlib import Path

    from llm_redact.detection.regex_rules import BUILTIN_RULES

    example = (Path(__file__).resolve().parent.parent / "config.example.toml").read_text()
    assert not re.search(r"^enabled = \[", example, flags=re.M), (
        "config.example.toml must not carry a LIVE enabled list — it would "
        "freeze copiers' rule set; keep it commented out"
    )
    listed = set(re.findall(r'^#\s+"([a-z_0-9]+)",$', example, flags=re.M))
    missing = [r.name for r in BUILTIN_RULES if r.name not in listed]
    assert not missing, f"config.example.toml commented enabled list is missing: {missing}"
