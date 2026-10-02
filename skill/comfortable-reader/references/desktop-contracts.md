# Desktop contracts and regression procedure

Read this file only when changing the reader application, its storage, pagination, annotations, or when the user explicitly requests a full reader regression. Book conversion alone does not authorize rewriting or reinstalling the app. Use isolated fixtures for implementation tests, and report installed-app evidence separately from browser previews and mocks.

## State and annotation invariants

- A stable book identifier owns its CFI/progression, page mode and annotations. Global theme and the workspace split tree belong to the session. Per-book readingPreferences own font/size/line-height/width; legacy session typography remains the fallback. Never key a note by a transient pane, iframe, computed screen page or window.
- Older progress records with no annotation field load as an empty list. Saving pagination must preserve unknown compatible annotation fields and existing marks.
- Original-text marks use an EPUB CFI range plus exact selected-text audit data. Runtime wrappers are ignored when resolving later CFIs. The embedded source EPUB bytes never change.
- New free notes and strokes use the actual visible content CFI plus normalized within-column coordinates (anchorVersion 2); preserve legacy progression records. Finishing an index must not move existing new notes. Their layer shares the EPUB canvas's logical size, origin and transform.
- Reading/selection mode keeps the annotation layer pointer-transparent. Hit testing is enabled only for explicit text, pen, eraser, edit, or drag modes.
- Deletion and undo are scoped to the current book. There is no unconfirmed destructive clear-all operation.

## Library and installation invariants

- A particular installed production build uses its own stable application identifier, progress store, and registered roots. The existing Windows 0.4 personal build uses `%APPDATA%\ComfortableReader\library-roots.json`; a portable fork or test build must discover its own identity and state root rather than importing this user's registry.
- Refresh scans that build's registered roots and any library the user deliberately linked for case-insensitive `.epub` files, follows links safely, deduplicates overlapping roots, and reports traversal/permission/invalid-ZIP errors. Do not assume a canonical Calibre path exists on another computer.
- Exclude `.caltrash` and recycle-bin trees. Report PDF/MOBI/AZW3/DOCX/TXT files as awaiting conversion; do not count them as loaded EPUBs.
- The library is complete only when `loadedCandidates == epubCandidates` and no registered root unexpectedly disappeared.
- Test builds use isolated state and identifiers. Never let a test fixture enter the production roots.

Bind the installed executable to its actual installer payload, not only a standalone EXE beside the installer. Tauri can patch its bundle-type marker while packaging (for example `UNK` to `NSS` for NSIS). If hashes differ, verify the exact changed bytes against current packaging code and record both hashes; a matching version or file size alone is insufficient. Preserve the executable's actual installed hash in runtime evidence.

## Required regression sequence

Run the synthetic checks first:

```powershell
python scripts/test_box_fidelity.py
python scripts/test_reader_pipeline.py
```

Then use an isolated long structured fixture and capture evidence for:

1. Auto, 3, 4 and 10 visible pages. Ten-page evidence needs at least ten real content pages; a short book with blank slots is insufficient. Check computed font size, ordinary line length, both edge pages, formulas, images, tables and algorithms.
2. Four-page single navigation: `1–4 → 2–5`. Four-page grouped navigation: `1–4 → 5–8`. Test backward single/group movement and a partial final group with real pages left-aligned and blank unused slots.
3. Two books in separate panes with different modes. Close/reopen and verify each CFI, mode and canvas is independent.
4. Auto and fixed-mode resize, maximize and full screen. Drive at least 60 width changes. In fixed mode, leftmost CFI, page range and total remain unchanged; no loading layer, clipped edge page or visible trembling appears.
5. A complete figure, long caption, bordered table, long algorithm/code, blockquote, display and boxed formula at boundaries. Preserve semantic units without cropping; long content flows at correct steps/rows/paragraphs, retaining headers and case ranges. Wide structures have readable selectable detail views.
6. Actual installed-app jump-to-page and internal links, not merely a visible button. Record the target text and CFI before/after.
7. Text marks combining highlight, bold, underline, wave, box, font, size and comment; navigate, restart, reopen in another pane, and compare source EPUB hash.
8. Free text note edit/drag and a multi-stroke five-point star; test pen width/color, eraser, undo, list deletion and jump-to-anchor through 3↔10 pages and 60 resize frames.
9. Load old progress/session records without annotations or readerFont; round-trip styled marks and drawings without loss. Reopen directly in dense mode at a spine boundary, including a preceding section whose font metrics differ before styling.
10. Explicit panel opening/closing, iframe keyboard focus, wheel pane selection, full-book search with ligatures, internal-link return, and font/theme round trips without stylesheet accumulation. Check actual response timing and indexing during rapid mode changes.

For each item record `installed_app`, `fixture_or_book_id`, `before`, `after`, `evidence_path`, and `status`. A browser screenshot or a mock IPC response must be marked as such and cannot be reported as installed-app verification.

For visual and interaction regressions, inspect text buttons separately from fixed-size icon buttons, check computed foreground/background colors after each theme change, and make the last control reachable in short viewports by actual panel scrolling and keyboard navigation. Save real screenshots together with element bounds, scroll dimensions, focus/ARIA state, and the interaction that produced them. A source patch or isolated CSS experiment remains pending until the affected browser and installed-app paths are retested; use the review breadth requested for the current task rather than fixing a reviewer count into every future task.

## Workspace continuity (0.6)

Keep up to 12 distinct books in stable slots. The split tree records relationships and ratios; focus and insufficient-space presentation must never overwrite it. Opening another book may split a local region after manual adjustment; resizing the outer window must not change topology. Closing a pane is not removing a book from the library. Duplicate views of one book are not supported yet.

The 0.7 tool rail owns a narrow, stable area outside the text. Expanding/collapsing its tools does not change book geometry. Panels may float when explicitly opened; docking reserves real space and reflows with a semantic anchor. Moving a floating panel does not move the books. Panel geometry, docking and UI scale live in session.surfaces; they must not reset book preferences or annotations. Temporarily clamp panels for a small viewport without overwriting their intended large-window geometry.

The compact shelf uses one panel for all books and currently-open books; it must not stack a second book-management overlay. Constrain shelf width, fade overflowing titles, and reveal low-frequency metadata only when needed. Check the actual native window and persisted geometry, not just the ability to resize a large panel. Surface state version 2 migrates the former oversized shelf to the compact default without resetting the workspace tree or personal records.

Full-screen window gestures use the blank safe top edge with no visible grab mark. A downward drag commits on release; cancelling must not change geometry. Restoration clamps the native window to the monitor work area, and close uses the existing save handler. Keep book-layout drag, reading-tool parking and OS-window actions separate; test the native action in addition to any mocked IPC tests.

Shelf dragging adds a book, never replaces another. Preview the complete candidate rectangles without changing live iframe layout; commit only on release. Preserve the shelf query/scroll position on cancellation. Edge drops split a local region, center drops exchange existing books, divider drops insert between groups. Empty space and insufficient-space fallbacks must retain all identities.

Page-turn targets live in book margins outside the EPUB/annotation canvas. Pointer hover never navigates; wheel gestures respect nested scroll owners, including their boundaries. Continuous reading keeps native scrolling. Text input, media controls and local scrolling keep their keyboard events. Progress dragging previews, then commits once. Test shelf dragging, cancel, panel resizing/docking, toolbar parking, sparse slots and restart in the real installed application.

## Resize and density rules

Read [dynamic-pagination-contract.md](dynamic-pagination-contract.md) before modifying layout. The hot resize path updates one compositor transform only. It does not call EPUB reflow, clear views, regenerate locations, change CFI, update headers, or show a loading layer. Fixed-mode resize is presentation-only.

The current effective font uses a calibrated starting formula:

```text
userFontPercent * sqrt(referencePages / (selectedPages * 1.2))
```

The 1.2 factor is not a universal comfort requirement. Compare 3 and 10 columns by actual computed size, line length and usable density. Auto scales during dragging, then may reflow after idle while retaining content. A character-location index is stable navigation labelled 阅读位置, not an approximate screen-page counter.

## Reporting and stopping

Report which checks were static, browser-only, mocked, or installed-app checks. If a required check cannot be performed, leave the status `awaiting_desktop_verification` and name the missing evidence. Do not alter production app state to make a regression appear green.
