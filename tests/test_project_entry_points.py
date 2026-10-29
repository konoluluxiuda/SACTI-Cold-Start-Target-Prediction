"""Check project naming and configuration selection without starting training."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from main import build_parser
from util.config import ModelConf


ROOT = Path(__file__).resolve().parents[1]


class ProjectEntryPointTests(unittest.TestCase):
    def test_default_configuration_is_a_released_sacti_pilot(self):
        with patch.dict(os.environ, {}, clear=True):
            args = build_parser().parse_args([])
        config = ROOT / args.config
        self.assertTrue(config.is_file())
        self.assertEqual(ModelConf(str(config))['model.name'], 'SACTI')

    def test_environment_and_explicit_configuration_selection(self):
        with patch.dict(os.environ, {'SACTI_CONFIG': 'environment.conf'}):
            self.assertEqual(build_parser().parse_args([]).config, 'environment.conf')
            self.assertEqual(build_parser().parse_args(['--config', 'explicit.conf']).config,
                             'explicit.conf')

    def test_help_uses_sacti_project_name(self):
        help_text = build_parser().format_help()
        self.assertIn('SACTI', help_text)
        self.assertNotIn('HDCTI', help_text)

    def test_shell_launcher_configuration_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            stub = Path(directory) / 'python'
            stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'],
                       CONDA_PREFIX=directory, SACTI_VERBOSE_TF='1')
            env.pop('SACTI_CONFIG', None)
            for config, extra, expected in (
                (None, [], 'configs/SACTI_tcmsuite_inner_pilot.conf'),
                ('environment.conf', [], 'environment.conf'),
                ('environment.conf', ['explicit.conf'], 'explicit.conf'),
            ):
                if config is not None:
                    env['SACTI_CONFIG'] = config
                result = subprocess.run([str(ROOT / 'run_sacti.sh'), *extra],
                                        cwd=ROOT, env=env, text=True, capture_output=True,
                                        check=True)
                self.assertEqual(result.stdout.splitlines(), ['main.py', '--config', expected])


if __name__ == '__main__':
    unittest.main()
