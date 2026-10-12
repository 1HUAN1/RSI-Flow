# Third-party notices

RSIFlow uses adapted third-party code. Local module names and comments have been
updated to RSIFlow; this does not change the origin or licensing of that code.

- The runtime in `RSIFlow/runtime/rsiflow` derives from
  [SIA](https://github.com/hexo-ai/sia), Copyright (c) 2025 Hexo.
  The original MIT license is preserved in [LICENSE](LICENSE).
- Task-agent components, generation templates, and validation utilities in
  `RSIFlow/upstream/RSIFlow_task` derive from
  [HarnessForge](https://github.com/mingju-c/HarnessForge), pinned upstream commit
  `05b3ecadb3c9a7a938f75129ea22b8f2b36cf289`. The renamed runtime adapters and seed
  bundle use these components. The upstream commit recorded in manifests denotes
  provenance; local files include RSIFlow adaptations and naming changes.
- Benchmark evaluators under `third_party/report_evaluators` retain their source
  documentation and license files. Codex build provenance is recorded separately
  in `third_party/codex-provenance.json`.

Keep these notices and the included license files when redistributing this release.
