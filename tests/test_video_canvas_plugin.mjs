import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const plugin = require("../integrations/video-canvas/provider-colab-studio.js");

function makeCtx(overrides = {}) {
  return {
    now: 1700000000,
    task: { id: 73 },
    model: { upstreamModel: "minimax-h3-ref2v" },
    channel: {
      settings: {
        mode: "fast8",
        video_vae: "dully",
        loras: "none",
        default_resolution: "720P",
        max_runtime_minutes: "60",
        lifecycle: "smart",
        retain_idle_minutes: "3",
      },
    },
    input: {
      prompt: "<Picture 1> 和 <Picture 2> 并肩走向镜头",
      images: ["fileRef://img-67", "fileRef://img-65"],
      aspect_ratio: "9:16",
      resolution: "720P",
      duration: 5,
      seed: 3916315536,
    },
    ...overrides,
  };
}

function runPipeline(ctxOverrides = {}) {
  const ctx = makeCtx(ctxOverrides);
  const prepReqs = plugin.buildPrepareRequests(ctx);
  const resps = prepReqs.map((_, idx) => ({
    status: 200,
    body: { id: idx === 0 ? "67" : String(65 - idx + 1) },
  }));
  const prepared = plugin.parsePrepareResponses(ctx, resps);
  const submit = plugin.buildSubmitRequest({ ...ctx, prepared });
  return { prepReqs, prepared, submit };
}

function testMeta() {
  assert.equal(plugin.meta.version, "1.4.0");
  assert.deepEqual(plugin.meta.channelSettings.mode.options, ["fast8", "refine8", "hq"]);
}

function testSeedBoundaries() {
  for (const fallbackVal of [undefined, null, ""]) {
    const { submit } = runPipeline({
      input: {
        prompt: "单图测试",
        images: ["fileRef://img-1"],
        aspect_ratio: "9:16",
        duration: 5,
        seed: fallbackVal,
      },
    });
    assert.equal(submit.json.params.seed, -1, `seed=${String(fallbackVal)} should fallback to -1`);
  }

  for (const [raw, expected] of [
    [-1, -1],
    [0, 0],
    [-0, 0],
    ["-1", -1],
    ["0", 0],
    [3916315536, 3916315536],
    ["3916315536", 3916315536],
    [4294967295, 4294967295],
    ["4294967295", 4294967295],
  ]) {
    const { submit } = runPipeline({
      input: {
        prompt: "单图测试",
        images: ["fileRef://img-1"],
        aspect_ratio: "9:16",
        duration: 5,
        seed: raw,
      },
    });
    assert.equal(submit.json.params.seed, expected);
  }

  for (const invalid of [" ", "1.8", 1.8, "-1.5", "bad", "1e5", -2, 4294967296, "9007199254740993", true]) {
    assert.throws(
      () =>
        runPipeline({
          input: {
            prompt: "单图测试",
            images: ["fileRef://img-1"],
            aspect_ratio: "9:16",
            duration: 5,
            seed: invalid,
          },
        }),
      /seed 需为 -1 或 0\.\.4294967295 的整数/,
      `Expected seed=${JSON.stringify(invalid)} to throw`,
    );
  }
}

function testDurationsAndDimensions() {
  const twoSec = runPipeline({
    input: {
      prompt: "单图 2s 测试",
      images: ["fileRef://img-1"],
      aspect_ratio: "9:16",
      duration: 2,
    },
  });
  assert.equal(twoSec.submit.json.params.length, 56);
  assert.equal(twoSec.submit.json.params.width, 480);
  assert.equal(twoSec.submit.json.params.height, 864);

  const landscape = runPipeline({
    input: {
      prompt: "16:9 横屏",
      images: ["fileRef://img-1"],
      aspect_ratio: "16:9",
      resolution: "720P",
      duration: 5,
    },
  });
  assert.equal(landscape.submit.json.params.width, 864);
  assert.equal(landscape.submit.json.params.height, 480);
  assert.equal(landscape.submit.json.params.length, 124);

  const fhd = runPipeline({
    input: {
      prompt: "1080P 竖屏",
      images: ["fileRef://img-1"],
      aspect_ratio: "9:16",
      resolution: "1080P",
      duration: 10,
    },
  });
  assert.equal(fhd.submit.json.params.width, 544);
  assert.equal(fhd.submit.json.params.height, 960);
  assert.equal(fhd.submit.json.params.length, 243);
}

function testRefAndI2VAndTwoPhaseDownload() {
  const baseline = makeCtx();
  const prepReqs = plugin.buildPrepareRequests(baseline);
  assert.equal(prepReqs.length, 2);
  assert.equal(prepReqs[0].path, "/api/assets");
  assert.equal(prepReqs[0].multipart.parts[0].fileRef, "fileRef://img-67");

  const prepared = plugin.parsePrepareResponses(baseline, [
    { status: 200, body: { data: { id: "67" } } },
    { status: 200, body: { data: { id: "65" } } },
  ]);
  const dullySubmit = plugin.buildSubmitRequest({ ...baseline, prepared });
  assert.equal(dullySubmit.method, "POST");
  assert.equal(dullySubmit.path, "/api/gpu/jobs");
  assert.equal(dullySubmit.headers["Idempotency-Key"], "canvas-73");
  assert.equal(dullySubmit.json.workflow_key, "minimax-h3-ref2v");
  assert.equal(dullySubmit.json.workflow_version, "1.4.0");
  assert.equal(dullySubmit.json.params.ref_1, "67");
  assert.equal(dullySubmit.json.params.ref_2, "65");
  assert.equal(
    dullySubmit.json.params.video_vae,
    "minimax_h3_video_vae_dully_seamless_v1_int8convrot.safetensors",
  );

  const officialSubmit = plugin.buildSubmitRequest({
    ...baseline,
    channel: {
      settings: {
        ...baseline.channel.settings,
        video_vae: "official",
      },
    },
    prepared,
  });
  assert.equal(officialSubmit.json.params.video_vae, "minimax_h3_video_vae_int8_convrot.safetensors");

  // Image limits
  assert.throws(
    () =>
      plugin.buildPrepareRequests(
        makeCtx({ input: { prompt: "x", images: [] } }),
      ),
    /参考图数量须在 1\.\.6/,
  );
  assert.throws(
    () =>
      plugin.buildPrepareRequests(
        makeCtx({ input: { prompt: "<Picture 1>", images: ["1", "2", "3", "4", "5", "6", "7"] } }),
      ),
    /参考图数量须在 1\.\.6/,
  );
  assert.throws(
    () =>
      plugin.buildPrepareRequests(
        makeCtx({
          model: { upstreamModel: "minimax-h3-i2v" },
          input: { prompt: "x", images: ["1", "2", "3"] },
        }),
      ),
    /参考图数量须在 1\.\.2/,
  );
  assert.throws(
    () =>
      plugin.buildPrepareRequests(
        makeCtx({ input: { prompt: "<Picture 2>", images: ["1"] } }),
      ),
    /Picture 引用越界/,
  );

  // I2V pipeline
  const i2vCtx = makeCtx({
    model: { upstreamModel: "minimax-h3-i2v" },
    input: {
      prompt: "<Picture 1> move gently",
      images: ["fileRef://first", "fileRef://last"],
      aspect_ratio: "9:16",
      duration: 5,
      seed: 42,
    },
  });
  const i2vPrep = plugin.parsePrepareResponses(i2vCtx, [
    { status: 200, body: { id: "asset-first" } },
    { status: 200, body: { id: "asset-last" } },
  ]);
  const i2vSubmit = plugin.buildSubmitRequest({ ...i2vCtx, prepared: i2vPrep });
  assert.equal(i2vSubmit.json.workflow_key, "minimax-h3-i2v");
  assert.equal(i2vSubmit.json.workflow_version, "1.1.0");
  assert.equal(i2vSubmit.json.params.first_frame, "asset-first");
  assert.equal(i2vSubmit.json.params.last_frame, "asset-last");
  assert.equal(i2vSubmit.json.params.steps, 8);
  assert.equal(i2vSubmit.json.params.seed, 42);

  // Submit response + query two-phase binary download
  const submitted = plugin.parseSubmitResponse(baseline, { status: 200, body: { job_id: "job-73" } });
  assert.equal(submitted.providerTaskId, "job-73");

  const pollReq = plugin.buildQueryRequest({ task: { providerTaskId: "job-73" } });
  assert.equal(pollReq.method, "GET");
  assert.equal(pollReq.path, "/api/gpu/jobs/job-73");

  const stage1 = plugin.parseQueryResponse({}, {
    status: 200,
    body: { state: "completed", output_asset_ids: ["out-video-99"] },
  });
  assert.deepEqual(stage1, {
    status: "running",
    progress: 99,
    state: { downloadAssetId: "out-video-99" },
  });

  const downloadReq = plugin.buildQueryRequest({ task: { providerTaskId: "job-73", state: stage1.state } });
  assert.equal(downloadReq.method, "GET");
  assert.equal(downloadReq.path, "/api/assets/out-video-99/download");
  assert.equal(downloadReq.responseType, "binary");

  const stage2 = plugin.parseQueryResponse({}, { asset: { id: "vc-asset-1" } });
  assert.deepEqual(stage2, {
    status: "succeeded",
    progress: 100,
    outputs: [{ type: "asset" }],
  });

  assert.equal(
    plugin.classifyError({}, { status: 500, body: { error: { code: "REAL_GPU_DISABLED", message: "off" } } }).class,
    "terminal",
  );
  assert.equal(
    plugin.classifyError({}, { status: 503, body: { error: { message: "busy" } } }).class,
    "retryable",
  );

  // Also test a 1080P two-ref Ref2V payload and a random seed payload for Colab manifest binding
  const ref2sCtx = makeCtx({
    input: {
      prompt: "<Picture 1> 挥手微笑",
      images: ["fileRef://img-67"],
      aspect_ratio: "9:16",
      resolution: "720P",
      duration: 2,
      seed: -1,
    },
  });
  const ref2sPrepared = plugin.parsePrepareResponses(ref2sCtx, [{ status: 200, body: { id: "67" } }]);
  const ref2sSubmit = plugin.buildSubmitRequest({ ...ref2sCtx, prepared: ref2sPrepared });

  const ref169Ctx = makeCtx({
    input: {
      prompt: "<Picture 1> 穿行于霓虹街头",
      images: ["fileRef://img-67"],
      aspect_ratio: "16:9",
      resolution: "1080P",
      duration: 5,
      seed: 0,
    },
  });
  const ref169Prepared = plugin.parsePrepareResponses(ref169Ctx, [{ status: 200, body: { id: "67" } }]);
  const ref169Submit = plugin.buildSubmitRequest({ ...ref169Ctx, prepared: ref169Prepared });

  return {
    dully: dullySubmit.json,
    official: officialSubmit.json,
    ref2s: ref2sSubmit.json,
    ref169: ref169Submit.json,
    i2v: i2vSubmit.json,
    pluginPath: path.resolve(__dirname, "../integrations/video-canvas/provider-colab-studio.js"),
    pluginBytes: fs.readFileSync(
      path.resolve(__dirname, "../integrations/video-canvas/provider-colab-studio.js"),
    ).length,
  };
}

testMeta();
testSeedBoundaries();
testDurationsAndDimensions();
const payloads = testRefAndI2VAndTwoPhaseDownload();

if (process.argv.includes("--emit-payloads")) {
  process.stdout.write(JSON.stringify(payloads));
} else {
  console.log("PASS: provider-colab-studio.js v1.4.0 contract tests");
}
