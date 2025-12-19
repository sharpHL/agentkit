import pytest
from click.testing import CliRunner
from pathlib import Path
from agentkit.cli.main import cli


@pytest.mark.integration
class TestCLI:
    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "AgentKit" in result.output

    def test_init_command(self, tmp_path):
        """Test agentkit init command"""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert Path(".env").exists()
            assert Path("examples/simple_agent.py").exists()

    def test_create_command(self, tmp_path):
        """Test agentkit create command"""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["create", "my_agent"])

            assert result.exit_code == 0
            assert Path("my_agent.py").exists()

            content = Path("my_agent.py").read_text()
            assert "MyAgentAgent" in content
