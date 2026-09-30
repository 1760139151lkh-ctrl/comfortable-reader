#!/usr/bin/env python3
"""Regression test for ChatGPT box, table, quote, code, and source fidelity."""

from __future__ import annotations

import tempfile
import re
import unittest
import zipfile
from pathlib import Path

from build_reader import MATH_BOX_CLASS, make_epub, render_markdown


SOURCE = r'''# 方框保真测试

\[
\boxed{\text{现实系统}\rightarrow\text{任务识别}}
\]

\[
\boxed{x+1}+\boxed{\text{第二个框}}
\]

> 预测得准，不等于知道为什么。

| 层次 | 核心问题 |
|---|---|
| 任务层 | 题目要回答什么？ |

```text
代码中的 \boxed{原样保留，不渲染}
```
'''


class BoxFidelityTest(unittest.TestCase):
    def test_boxed_math_and_bordered_blocks_survive_epub(self) -> None:
        warnings: list[str] = []
        fragment = render_markdown(SOURCE, warnings)
        self.assertEqual(warnings, [])
        self.assertEqual(fragment.count(MATH_BOX_CLASS), 3)
        self.assertIn("<blockquote>", fragment)
        self.assertIn("<table>", fragment)
        self.assertIn("<pre>", fragment)

        with tempfile.TemporaryDirectory(prefix="comfortable-reader-box-test-") as temp_dir:
            output = Path(temp_dir) / "box-fidelity.epub"
            info = make_epub(
                output,
                "方框保真测试",
                "Comfortable Reader",
                fragment,
                "box-fidelity-regression",
                SOURCE,
                "source.md",
            )
            self.assertEqual(info["boxed_math_regions"], 3)
            with zipfile.ZipFile(output) as archive:
                content = archive.read("OEBPS/content.xhtml").decode("utf-8")
                stylesheet = archive.read("OEBPS/style.css").decode("utf-8")
                embedded = archive.read("OEBPS/original/source.md").decode("utf-8")
            self.assertEqual(content.count(MATH_BOX_CLASS), 3)
            self.assertIn('menclose[notation~="box"]', stylesheet)
            rules={}
            for selectors, body in re.findall(r'([^{}]+)\{([^{}]+)\}',stylesheet):
                for selector in selectors.split(','):
                    declarations=rules.setdefault(selector.strip(),{})
                    for declaration in body.split(';'):
                        if ':' in declaration:
                            name,value=declaration.split(':',1);declarations[name.strip()]=value.strip()
            quote_border=rules['blockquote'].get('border-inline-start',rules['blockquote'].get('border-left',''))
            self.assertIn('solid',quote_border)
            self.assertIn('solid',rules['th']['border'])
            self.assertIn('solid',rules['td']['border'])
            self.assertEqual(embedded, SOURCE)


if __name__ == "__main__":
    unittest.main()
