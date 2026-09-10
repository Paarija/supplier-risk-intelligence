from supplyshield.supplier_matching import article_matches_supplier, normalize_name


def test_company_suffixes_are_removed():
    assert normalize_name("Orion Components Ltd.") == "orion components"


def test_alias_matches_article():
    assert article_matches_supplier(
        "Orion Electronics faces a port delay.",
        "Orion Components",
        "Shenzhen, China",
        aliases="Orion Electronics|Orion Components Ltd",
    )


def test_location_can_match_article():
    assert article_matches_supplier(
        "A major factory shutdown was reported in Houston.",
        "Delta Chemicals",
        "Houston",
    )
