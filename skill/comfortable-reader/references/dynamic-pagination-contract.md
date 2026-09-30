# Reflow, navigation and annotation contracts

Read before implementing or validating layout. Keep stable content identity separate from screen geometry. Implementation constants are calibrated choices, not universal comfort requirements.

## Content positions and pages

Persist the CFI and per-book mode. Screen pages depend on viewport, fonts and layout; a raw page number is not a durable location.

For multi-section books a stable character-location index is an acceptable final model when labelled **阅读位置**, with percent. It is neither a count nor an estimate of screen pages. The current reader builds roughly 1,000-character locations once per opened book. Reflow does not invalidate this text index. Guard in-flight builds so rapid mode changes cannot continually restart them; reject results for a different book generation.

Single-section screen-page counts may use exact rendition data. A UI promising exact page jumps must use the same model for labels and jumps. Exact multi-section page counts require measured section layouts, not extrapolation from text length.

For the reader's one-section books, screen-page jumps use the measured logical column width and current scroll position, not the character index. Check fractional-width columns: Chromium scroll quantization plus a floor operation can mislabel page 3 as page 2. Snap the page ordinal to the measured column grid and test several interior pages and the end in Auto/3/4/10, not just page 1.

## Geometry and navigation

With logical width W and n columns, use pageWidth = W/n, columnWidth = pageWidth-gap, singleStep = pageWidth, groupStep = n*pageWidth. Clamp the gap to leave usable text width. epub.js includes half a gap at each outer edge; adding that twice clips the viewport. Keep a small safe inset for fractional DPI. Check ink bounds as well as element boxes; long headings/NBSPs need emergency wrapping.

Left/Right and deliberate wheel steps move one column (1–4 → 2–5). PageUp/PageDown and visible group controls move a screen (1–4 → 5–8). Controls can live in the stable toolbar rather than over text. Route actions and iframe focus to the intended pane. Keep adjacent spine sections available for continuous chapter transitions.

Only after the final spine section mounts, reserve up to n-1 blank tail slots. A partial final group has real columns followed by blanks; don't repeat earlier content to fill it or insert a spacer after intermediate sections.

Serialize reflow, navigation and jumps per pane; coalesce queued reflows and cancel stale operations when the book changes. Pending reflow must not undo a later TOC/search jump. Resolve targets to real text/element CFIs, including image elements. Wait for fonts/assets/dimensions before restoring; suppress transient relocated events, then save the settled location.

Apply initial font metrics and the owned reading stylesheet **before the first iframe size measurement**, not only in a post-display content hook. In the continuous manager, a prepended section shrinking from its temporary default-font width can move the origin several chapters. The tested implementation seeds styles in the spine serialize hook, then maintains a single owned stylesheet for live updates. Keep head-style seeding separate from any necessary display-only body wrappers: those must preserve source EPUB bytes and text, use the same ignored wrapper class as annotations, and pass old/new character-level CFI resolution and restart tests. Verify reopening directly in dense modes at chapter boundaries.

After intentional reflow, the target may lie lower in its new whole column, so the first visible CFI can change slightly. Require the target content to remain visible; don't demand identical old line breaks. Fixed-canvas resize and same-layout restart should retain the corresponding CFI.

## Density, fonts and resize

Increasing n at constant font creates narrow strips. The approximation pageCapacity ∝ pageArea/fontSize² suggests a square-root adjustment. The current reader retains sqrt(referenceColumns/(selectedColumns*1.2)); the factor 1.2 and “20% more text” are not universal design laws. Verify actual line lengths, glyph size, mathematical structure and whitespace. Ten columns are an explicit overview mode, not the default for long reading.

Book CSS uses relative font sizes and reader typography roles. Publisher mode removes reader body-family overrides; math/code keep their own roles. Replace a single theme sheet instead of appending rules. Inspect computed styles and switch-back behavior.

During resize gestures update one canvas transform per animation frame. Use one resize owner and numeric logical dimensions (or detach epub.js's competing listener). The hot path must not rebuild views/locations, call display or show a loading cover.

In fixed-count mode retain logical W0,H0 across native resize/fullscreen/grid changes and use scale = min((viewportWidth-2*inset)/W0, (viewportHeight-2*inset)/H0). Pin the top-left origin; changing translation and scale together causes trembling. Deliberate mode/font changes may rebuild the canvas. Auto scales while dragging, then may repaginate after an idle debounce while restoring the content anchor; don't lock Auto to an old width forever.

## Fragmentation by semantics

Use fixed-height columns with column-fill:auto. Paragraphs flow normally. Preserve intact images, individual display equations, short tables and short figure/caption units when they fit; measure math labels/ink as well as outer boxes.

Long algorithms/code flow at complete instruction or line boundaries, preserving indentation, comments, numbers and short branch groups. Long tables can flow by rows with repeated headers. Long captions continue at readable size and stay semantically associated with their image. Do not scale these entire structures to their total height.

Width fitting is different from height fitting. A wide relation table/formula may have a compact overview plus an explicit selectable HTML/MathML detail view at reading size. Preserve the scaled layout footprint. A tall case brace must not control source-line sorting; use text baselines and check its actual instruction range.

## Annotations

Text marks use a CFI range, exact selected-text audit, style/comment/time. Creation and resolution ignore runtime wrapper classes; marks never change EPUB bytes.

New free notes and strokes anchor to the actual visible text/element CFI of their logical column, plus normalized within-column coordinates. The current anchorVersion=2 distinguishes these from legacy progression records. Preserve older records and use progression as a compatible fallback, not an authority that moves new notes when indexing completes. Re-resolve the content anchor after reflow.

The overlay and EPUB host share logical size, top-left inset and exactly the same transform. Fixed resize only changes that transform. Keep overlays pointer-transparent in reading/selection mode. Saves preserve other books and existing compatible fields; old records without annotations or readerFont still load.

## Evidence

Use enough actual text for Auto/3/4/10 full columns. Check fonts/edges, cross-spine single/group movement, partial tail, two independent panes, TOC/search/internal links/return, close/reopen and native restart. Test at least 60 rapid resizes plus fullscreen/restore: fixed mode retains canvas/CFI; Auto settles and keeps the target content visible.

Include long algorithms/captions, wide formulas/tables, dense inline math and long narrow headings. Exercise text marks, free notes and drawings across indexing, mode changes, resize and reload; compare EPUB hashes.

Label evidence as structural, headless browser, mocked IPC, isolated native build or installed application. Record artifact hash, code version, viewport and device scale. Confirm the final stored record/path and scan counts in the installed application. Changed rendering rules require renewed evidence for affected cases, not repetition of unrelated passed checks.


## Ink geometry compatibility (reader 0.4)

Semantic anchors remain anchorVersion 2. Drawing records may also contain inkGeometry with the creation column width, logical page height and drawing group. The group scales uniformly around its shared center when columns change, so strokes do not stretch independently in x/y. Original points and CFIs remain intact. Legacy records have no recorded creation aspect ratio: the upgrade records the current layout as an explicit legacy-current-layout reference, not as a reconstruction of the original drawing conditions. Keep that provenance and the original records; do not claim historical geometry was recovered.

Free text notes retain readable widths and are fitted inside the visible canvas after measurement; this display adjustment does not rewrite their semantic anchor. Dragging begins from the rendered position to avoid a jump after boundary fitting. Recheck both appearance and stored data through mode changes; unchanged JSON alone does not prove an undistorted drawing or an unclipped note.

For reflowable EPUBs, 0.4 records sourceSha256 with saved progress. If later bytes differ, ordinary reading remains available while old CFI/annotations are held without being applied to the new DOM or overwritten. The header and notes panel show that the edition needs review. Restore the exact original EPUB and reopen to resume it, or perform an explicit, evidenced edition mapping. A legacy record without a digest can only establish its current bytes on first opening; this does not recover an unknown historical edition. No fuzzy matching is silently treated as a verified migration.

## Paged and scroll views

The 0.5 integration keeps readingMode separately from pageMode and theme. A mode switch reuses the book identity, annotations and semantic CFI; it neither imports another book nor restarts an activity. A new web reader may begin in scroll mode, while installed preferences take precedence. Resizing never changes the chosen reading direction.

Free notes and ink in scroll mode must move with their content anchors. Optional contentPlacement records the viewport-relative anchor at creation, so a later scroll or reflow can project the note from actual content geometry. Keep original points and legacy provenance. Test the visible drawing and note movement, not merely unchanged JSON.

For streamed books, check the canonical identity of declared EPUB members against the local EPUB's actual member identities before moving personal annotations across platforms. A publisher-provided whole-EPUB hash alone does not prove that separately supplied XHTML matches it. Every fetched member must match its bound size and hash. Keep older records without a comparable content identity until they can be checked; do not silently treat a guessed mapping as verified.
