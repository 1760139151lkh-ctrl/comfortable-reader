# Referenced ChatGPT Conversation Fidelity

Use this procedure whenever the user supplies a `chatgpt-conversation://...` reference together with a locator screenshot or asks to reproduce one ChatGPT reply exactly.

## Select the exact message

1. Treat the screenshot only as a locator. Read the referenced conversation's structured turns as the source of truth.
2. Match the visible conversation title, preceding user message, and a distinctive visible sentence from the assistant reply.
3. Select exactly one assistant turn. Extract its complete `content.text`, from its first character through its last character, even when most of it is outside the screenshot.
4. Do not include the preceding user message, app chrome, citations panel, timestamps, or other UI unless requested.
5. If several turns match, or the structured conversation is unavailable, stop and request a stronger anchor or the conversation reference. A screenshot alone cannot recover hidden text.

## Preserve the source

- Save the extracted text directly as UTF-8 Markdown with LF newlines. Use `apply_patch` or another Unicode-safe file API; never route Chinese source text through a shell argument, console pipe, OCR, or clipboard transformation.
- Do not summarize, rewrite, translate, correct, normalize, or add a body title.
- Preserve every heading level, blank line, horizontal rule, table separator, blockquote marker, list marker and number, code fence language, inline-code backtick, emphasis marker, link, full-width punctuation, curly quote, arrow, mathematical delimiter, backslash, and Unicode symbol.
- Use `--title`, `--author`, and `--source-reference` for metadata. Metadata must not be inserted into the reproduced body.
- Reject U+FFFD (`�`) or NUL characters. If mojibake is already present in the structured source, report it instead of guessing the intended characters.

## Render without losing recoverability

- The builder converts ChatGPT-style `\[...\]`, `\(...\)`, and `$$...$$` math outside code spans/fences to validated MathML. The normal CLI rejects unrendered mathematics. An explicitly accepted diagnostic draft may use `--allow-math-fallback`; that draft cannot be imported as the finished book.
- Treat `\\boxed{...}` as a visible formatting contract, not merely MathML semantics. Preserve it as `<menclose notation="box" class="comfortable-math-box">` and ship the CSS border fallback because Chromium/WebView2 can display the contents of `menclose` without drawing its outline.
- The EPUB embeds the untouched input-file bytes as `OEBPS/original/source.md` (or the matching extension), including any original BOM/line endings. Rendered HTML may reflow, but the original must remain byte-for-byte recoverable. This byte check is separate from checking the complete visible reply.
- Keep tables as bordered tables, blockquotes as visibly distinguished quote boxes, fenced code as bordered/preformatted blocks, and headings in the table of contents. A structurally present element whose intended outline is invisible is a fidelity failure.

## Validate before delivery

1. Require `replacement_characters: 0`, `warnings: []`, and matching `source_sha256` values in `fidelity` and `epub`.
2. Confirm `rendered_mathml_formulas` equals the sum of block and inline math regions for a normal ChatGPT Markdown source.
3. Count `\\boxed{...}` regions in the selected structured reply and compare them with `boxed_math_regions`. Confirm the generated XHTML has the same number of `comfortable-math-box` markers and the stylesheet contains the `menclose[notation~="box"]` fallback.
4. Confirm the EPUB validator compared the embedded source bytes with the input bytes.
5. Open the EPUB in the Comfortable Reader desktop app and visually inspect the beginning, every distinct box style, a table, block and inline formulas, a code block when present, and the final paragraph. Source/XML counts alone do not prove visible borders.
6. Verify four-page landscape view and one-page `Left`/`Right` sliding.
7. Rerun the exact import once; it must return `already_present` with the same book ID.
