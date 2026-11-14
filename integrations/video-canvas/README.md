# Video Canvas Plugin (`colab-gpu-studio` v1.4.0)

`provider-colab-studio.js` (SHA256: `17e45384539e8ef1c6371cc7bcbfccaa30bb0990d5a6ba8335818cc5ed6fc986`) implements the `video-canvas` plugin v1 contract targeting `colab-gpu-studio` (`minimax-h3-ref2v@1.4.0` and `minimax-h3-i2v@1.1.0`).

## Changes in `1.4.0`

- **Strict `seed` validation**:
  - `undefined`, `null`, and `""` resolve to `-1` (random).
  - Non-empty strings must match `/^-?\d+$/` (rejects `" "`, `"1.8"`, `"1e5"`, `"bad"`).
  - Parsed integer must be within `[-1, 4294967295]`; invalid values throw explicitly instead of truncating or coercing to `0`.
- **Honest Ref2V `mode` options**: keeps `["fast8", "refine8", "hq"]`.
- **Physical frame quantization (`24fps`, `22 + 17k` rule)**:
  - `2s` -> `56` frames (`≈ 2.33s @ 24fps`)
  - `5s` -> `124` frames (`≈ 5.17s @ 24fps`)
  - `10s` -> `243` frames (`≈ 10.125s @ 24fps`)
  - `15s` -> `362` frames (`≈ 15.08s @ 24fps`)
- **Two-phase completion download**: polling `completed` transitions to `{ status: "running", progress: 99, state: { downloadAssetId } }`, followed by a `responseType: "binary"` `GET /api/assets/{id}/download` step returning `{ status: "succeeded", outputs: [{ type: "asset" }] }`.

## Zero-GPU Contract Tests

```bash
node tests/test_video_canvas_plugin.mjs
pytest tests/test_video_canvas_integration.py
```

`tests/test_video_canvas_integration.py` invokes `node tests/test_video_canvas_plugin.mjs --emit-payloads` and passes the real generated `params` directly through `app.gpu.manifest.resolve_parameters` and `bind_parameters` without GPU hardware or outbound network.
