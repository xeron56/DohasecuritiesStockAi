"""CLI branding and compatibility entry-point checks."""

from pathlib import Path

from typer.testing import CliRunner

import cli.main as cli_main


def test_cli_help_uses_dohasecurities_brand():
    result = CliRunner().invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0
    assert "Doha Securities Stock AI" in result.output
    assert "DSE Multi-Agent Stock Analysis" in result.output


def test_product_command_and_legacy_alias_share_the_cli_entrypoint():
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = project_file.read_text(encoding="utf-8")

    assert 'name = "dohasecuritiesstockai"' in project
    assert 'DohasecuritiesStockAi = "cli.main:app"' in project
    assert 'dohasecuritiesstockai = "cli.main:app"' in project
    assert 'tradingagents = "cli.main:app"' in project
    assert (
        'dohasecuritiesstockai-api = "dohasecuritiesstockai.api.__main__:main"'
        in project
    )


def test_python_package_uses_dohasecuritiesstockai_directory():
    project_root = Path(__file__).resolve().parents[1]

    assert (project_root / "dohasecuritiesstockai" / "__init__.py").is_file()
    assert not (project_root / "tradingagents").exists()
