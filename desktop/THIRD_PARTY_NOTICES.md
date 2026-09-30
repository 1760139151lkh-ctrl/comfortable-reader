# Third-party notices

The application and build dependency inventory is in `licenses/dependencies.json`, with actual versions, declared licenses, retained notices and source locations. It includes build-only dependencies; their platform executables are not bundled with the reader. Upstream author and copyright notices remain intact.

The unmodified MPL-2.0 `selectors` 0.36.1 source is supplied in `licenses/source/selectors-0.36.1.crate` (SHA-256 `c5d9c0c92a92d33f08817311cf3f2c29a3538a8240e94a6a3c622ce652d7e00c`). It is also available from https://crates.io/crates/selectors/0.36.1. The covered source remains under MPL-2.0, whose full text is included. Other parts retain their respective licenses; a project-level license does not replace them.

`marks-pane` identifies MIT in both its original and fork package declarations. The accompanying notice explicitly records the origin and assembled standard MIT text rather than claiming an upstream LICENSE file existed. `path-webpack` identifies its origin as Node.js v7.0.0's path implementation with Win32 portions removed; that upstream license and package attribution are retained.

## PDF viewing dependencies

PDF.js 5.6.205 is bundled for offline rendering. Copyright Mozilla Foundation and contributors; Apache License 2.0. The complete license is in public/vendor/pdfjs/LICENSE. Standard PDF font licenses are retained in public/vendor/pdfjs/standard_fonts/LICENSE_FOXIT and LICENSE_LIBERATION. No system or cloud font directory was copied. Upstream: https://github.com/mozilla/pdf.js.

PDF documents are input data. The viewer does not instantiate PDF scripting/form action support; isEvalSupported=false, useWasm=false and enableXfa=false. Rendering is on demand and the document worker is destroyed when the view closes. Original PDF bytes remain unchanged.

The JavaScript OpenJPEG decoder fallback and its OpenJPEG/PDF.js licenses are retained in public/vendor/pdfjs/wasm. This viewer uses that local non-WASM decoder when a source PDF contains JPEG 2000 images. It does not load remote decoder code.
