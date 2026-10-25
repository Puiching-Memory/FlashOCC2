"""Exercise the trust boundary from GHSA-x9r9-c232-4q39 without network access."""

import unittest
from unittest.mock import patch

from transformers.generation import utils


class CustomGenerationTrustTest(unittest.TestCase):
    def test_declining_remote_code_does_not_download_module(self):
        # A repository contains custom code, but the caller explicitly refuses it.
        with (
            patch.object(utils, "has_file", return_value=True, create=True),
            patch.object(utils, "get_cached_module_file") as download,
            patch.object(utils, "check_python_requirements") as requirements,
            patch.object(utils, "get_class_in_module") as load,
        ):
            with self.assertRaises(ValueError):
                utils.GenerationMixin.load_custom_generate(
                    object(), "untrusted/model", trust_remote_code=False
                )
            download.assert_not_called()
            requirements.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
