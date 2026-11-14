/**
 * Colab GPU Studio · 视频生成插件（契约 v1，版本 1.4.0）。
 *
 * 对接 colab-gpu-studio (FastAPI)，调度 Google Colab L4 / A100 GPU 生成有声视频：
 *   1. 准备阶段（buildPrepareRequests）：将参考图 multipart 流式上传到 POST /api/assets，拿到 Studio 素材 ID；
 *   2. 提交阶段（buildSubmitRequest）：调用 POST /api/gpu/jobs 创建异步任务（支持 smart 短暂空闲保留热机复用、strict seed 整数控制与 2–15s 物理帧档位）；
 *   3. 轮询阶段（buildQueryRequest）：轮询 GET /api/gpu/jobs/{id}；当任务 completed 后，下一轮以 responseType: "binary"
 *      请求 GET /api/assets/{asset_id}/download，由宿主携带渠道鉴权直接流式落库成视频素材（{ type: "asset" }）。
 *
 * 物理帧量化说明（24fps，17k + 5）：
 *   2s -> 56 帧（约 2.33s @ 24fps）
 *   5s -> 124 帧（约 5.17s @ 24fps）
 *   10s -> 243 帧（约 10.125s @ 24fps）
 *   15s -> 362 帧（约 15.08s @ 24fps）
 */
module.exports = {
  meta: {
    apiVersion: 1,
    key: "colab-gpu-studio",
    name: "Colab GPU Studio · 视频",
    version: "1.4.0",
    vendor: "colab-gpu-studio",
    description: "通过 Colab GPU Studio 调度 Google Colab L4/A100 GPU 生成有声视频（支持智能热机连拍、Ref2V 模式/VAE/LoRA、strict seed [-1..4294967295] 及 2–15s 物理帧档位：2s=56 帧/5s=124 帧）",
    auth: { type: "bearer" },
    endpoints: {
      video: { mode: "async", method: "POST", path: "/api/gpu/jobs" }
    },
    channelSettings: {
      mode: {
        type: "enum",
        label: "Ref2V 采样模式",
        description: "仅 Ref2V 生效；fast8=8步快采，refine8=8步后1.2x高清放大，hq=20步高质",
        options: ["fast8", "refine8", "hq"],
        default: "fast8"
      },
      video_vae: {
        type: "enum",
        label: "Ref2V 视频 VAE",
        description: "仅 Ref2V 生效；dully=2D/平滑默认VAE，official=官方原版VAE",
        options: ["dully", "official"],
        default: "dully"
      },
      loras: {
        type: "enum",
        label: "Ref2V 附加 LoRA",
        description: "仅 Ref2V 生效；human-0.4=better-human-motion (0.4)",
        options: ["none", "human-0.4"],
        default: "none"
      },
      default_resolution: {
        type: "enum",
        label: "缺省分辨率档位",
        description: "输入未指定 clarity/resolution 时使用（Ref2V 9:16 对应 480x864 / 544x960 / 576x1024）",
        options: ["720P", "1080P", "4K"],
        default: "720P"
      },
      max_runtime_minutes: {
        type: "enum",
        label: "Ref2V 超时上限（分钟）",
        description: "仅 Ref2V 生效；提交 execution_policy.max_runtime_minutes（服务端按配置封顶）",
        options: ["30", "60", "90"],
        default: "60"
      },
      lifecycle: {
        type: "enum",
        label: "运行策略",
        description: "smart：跑完短暂保留热机等下一镜；save：跑完立刻销毁；batch：仅队列有排队任务时复用",
        options: ["smart", "save", "batch"],
        default: "smart"
      },
      retain_idle_minutes: {
        type: "enum",
        label: "空闲保留时长（分钟，仅 smart 生效）",
        options: ["2", "3", "5", "10", "15"],
        default: "3"
      }
    },
    poll: { firstDelay: 10, interval: 8, maxInterval: 20, jitter: 0.1 }
  },

  buildPrepareRequests: function (ctx) {
    var input = ctx.input || {};
    var images = collectImages(input);
    var upstream = (ctx.model && ctx.model.upstreamModel) ? String(ctx.model.upstreamModel) : "minimax-h3-i2v";
    var isRef2v = upstream.indexOf("ref2v") >= 0;
    var maxImages = isRef2v ? 6 : 2;
    checkImages(images, maxImages);
    if (isRef2v) {
      normalizeRefPrompt(input.prompt || "", images.length);
    }
    var limit = images.length;
    var reqs = [];
    for (var i = 0; i < limit; i++) {
      reqs.push({
        method: "POST",
        path: "/api/assets",
        multipart: {
          fields: { project_id: "default" },
          parts: [{ name: "file", fileRef: images[i] }]
        },
        timeout: 60
      });
    }
    return reqs;
  },

  parsePrepareResponses: function (ctx, resps) {
    var assetIds = [];
    var list = isArray(resps) ? resps : [];
    for (var i = 0; i < list.length; i++) {
      var body = (list[i] && list[i].body) ? list[i].body : {};
      var data = body.data || body;
      var id = data.id || data.asset_id;
      if (!id) {
        throw new Error("上传参考图到 Colab GPU Studio 失败：未返回素材 ID");
      }
      assetIds.push(String(id));
    }
    return { assetIds: assetIds };
  },

  buildSubmitRequest: function (ctx) {
    var input = ctx.input || {};
    var upstream = (ctx.model && ctx.model.upstreamModel) ? String(ctx.model.upstreamModel) : "minimax-h3-i2v";
    var isRef2v = upstream.indexOf("ref2v") >= 0;

    var settings = (ctx.channel && ctx.channel.settings) ? ctx.channel.settings : {};
    var expectedImages = collectImages(input);
    var assetIds = (ctx.prepared && isArray(ctx.prepared.assetIds)) ? ctx.prepared.assetIds : [];
    checkImages(assetIds, isRef2v ? 6 : 2);
    if (assetIds.length !== expectedImages.length) {
      throw new Error("参考图上传数量与输入不一致：期望 " + expectedImages.length + "，实际 " + assetIds.length);
    }

    var resolutionTier = input.resolution ? input.resolution : pickSetting(settings.default_resolution, ["720P", "1080P", "4K"], "720P");
    var dims = resolveVideoDimensions(input.aspect_ratio, resolutionTier, isRef2v);
    var length = resolveFrameLength(input.duration);
    var prompt = input.prompt ? String(input.prompt) : "Animate the reference image with gentle natural motion.";
    if (isRef2v) {
      prompt = normalizeRefPrompt(prompt, assetIds.length);
    } else if (prompt) {
      prompt = prompt.replace(/图片\s*[0-9０-９]+|<\s*picture\s+[0-9]+\s*>/gi, "the scene");
    }
    var seed = parseSeed(input.seed);

    var params = {
      prompt: prompt,
      width: dims.width,
      height: dims.height,
      length: length,
      seed: seed
    };

    var workflowKey = isRef2v ? "minimax-h3-ref2v" : "minimax-h3-i2v";
    var workflowVersion = isRef2v ? "1.4.0" : "1.1.0";

    var loras = [];
    if (isRef2v) {
      params.mode = pickSetting(settings.mode, ["fast8", "refine8", "hq"], "fast8");
      var vaeAlias = pickSetting(settings.video_vae, ["dully", "official"], "dully");
      params.video_vae = vaeAlias === "official"
        ? "minimax_h3_video_vae_int8_convrot.safetensors"
        : "minimax_h3_video_vae_dully_seamless_v1_int8convrot.safetensors";
      if (pickSetting(settings.loras, ["none", "human-0.4"], "none") === "human-0.4") {
        loras.push({ id: "better-human-motion", strength: 0.4 });
      }
      for (var r = 0; r < assetIds.length && r < 6; r++) {
        params["ref_" + (r + 1)] = assetIds[r];
      }
    } else {
      params.first_frame = assetIds[0];
      if (assetIds.length > 1) {
        params.last_frame = assetIds[1];
      }
      params.steps = input.steps ? Number(input.steps) : 8;
    }

    var taskId = (ctx.task && ctx.task.id) ? ("canvas-" + ctx.task.id) : ("canvas-" + ctx.now);
    var provider = settings.provider ? String(settings.provider) : "colab";
    var lifecycle = settings.lifecycle ? String(settings.lifecycle) : "smart";
    var idleMinutes = parseInt(String(settings.retain_idle_minutes || "3"), 10);
    if (!idleMinutes || isNaN(idleMinutes) || idleMinutes < 1) {
      idleMinutes = 3;
    } else if (idleMinutes > 30) {
      idleMinutes = 30;
    }
    var maxRuntime = isRef2v ? Number(pickSetting(settings.max_runtime_minutes, ["30", "60", "90"], "60")) : 30;

    var policy = {
      provider: provider,
      lifecycle: lifecycle,
      max_runtime_minutes: maxRuntime
    };
    if (lifecycle === "smart" || lifecycle === "retain") {
      policy.retain_idle_minutes = idleMinutes;
    }

    var body = {
      project_id: "default",
      model_id: "minimax-h3",
      workflow_key: workflowKey,
      workflow_version: workflowVersion,
      params: params,
      loras: loras,
      execution_policy: policy
    };

    return {
      method: "POST",
      path: "/api/gpu/jobs",
      headers: {
        "Idempotency-Key": taskId
      },
      json: body,
      timeout: 60
    };
  },

  parseSubmitResponse: function (ctx, resp) {
    var body = resp.body || {};
    if (body.error) {
      var err = body.error || {};
      return {
        immediate: {
          status: "failed",
          error: {
            class: "terminal",
            code: err.code ? String(err.code) : "submit_error",
            message: String(err.message || body.message || "Colab GPU Studio 提交任务失败")
          }
        }
      };
    }

    var data = body.data || body;
    var jobId = data.job_id || data.id;
    if (!jobId) {
      throw new Error("Colab GPU Studio 未返回有效的任务 ID (job_id)");
    }

    return {
      providerTaskId: String(jobId)
    };
  },

  buildQueryRequest: function (ctx) {
    var state = (ctx.task && ctx.task.state) ? ctx.task.state : null;
    if (state && state.downloadAssetId) {
      return {
        method: "GET",
        path: "/api/assets/" + encodeURIComponent(state.downloadAssetId) + "/download",
        responseType: "binary",
        timeout: 120
      };
    }

    var jobId = ctx.task ? ctx.task.providerTaskId : "";
    if (!jobId) {
      throw new Error("缺少 providerTaskId，无法轮询任务状态");
    }

    return {
      method: "GET",
      path: "/api/gpu/jobs/" + encodeURIComponent(jobId),
      timeout: 30
    };
  },

  parseQueryResponse: function (ctx, resp) {
    if (resp.asset && resp.asset.id) {
      return {
        status: "succeeded",
        progress: 100,
        outputs: [{ type: "asset" }]
      };
    }

    var body = resp.body || {};
    var data = body.data || body;
    var st = String(data.state || data.status || "").toLowerCase();

    if (st === "completed" || st === "succeeded" || st === "success") {
      var assetIds = isArray(data.output_asset_ids) ? data.output_asset_ids : [];
      if (assetIds.length === 0) {
        return {
          status: "failed",
          error: {
            class: "terminal",
            code: "empty_output",
            message: "Colab 任务已完成但未生成视频产物"
          }
        };
      }
      return {
        status: "running",
        progress: 99,
        state: { downloadAssetId: String(assetIds[0]) }
      };
    }

    if (st === "failed" || stateIsFailed(st)) {
      return {
        status: "failed",
        error: {
          class: "terminal",
          code: String(data.error_code || "execution_failed"),
          message: String(data.error_message || (data.error && data.error.message) || "Colab GPU 任务执行失败")
        }
      };
    }

    if (st === "queued" || st === "validating" || st === "waiting_for_capacity") {
      return {
        status: "queued",
        progress: 10
      };
    }

    var progress = 60;
    if (st === "allocating") progress = 25;
    else if (st === "preparing") progress = 45;
    else if (st === "cleaning_up") progress = 92;

    return {
      status: "running",
      progress: progress
    };
  },

  classifyError: function (ctx, resp) {
    var body = resp.body || {};
    var err = body.error || {};
    var code = err.code ? String(err.code) : "";
    var msg = String(err.message || body.message || (resp.text || "").slice(0, 300) || ("HTTP " + resp.status));

    if (resp.status === 429 || resp.status >= 500) {
      if (code === "REAL_GPU_DISABLED" || code === "GOOGLE_ACCOUNT_ERROR" || code === "REAL_LIFECYCLE_UNVERIFIED") {
        return { class: "terminal", code: code, message: msg };
      }
      return { class: "retryable", code: code || String(resp.status), message: msg };
    }
    return { class: "terminal", code: code || String(resp.status), message: msg };
  }
};

function parseSeed(raw) {
  if (raw === undefined || raw === null || raw === "") {
    return -1;
  }
  var n;
  if (typeof raw === "number") {
    n = raw;
  } else if (typeof raw === "string") {
    if (!/^-?\d+$/.test(raw)) {
      throw new Error("seed 需为 -1 或 0..4294967295 的整数: " + JSON.stringify(raw));
    }
    n = Number(raw);
  } else {
    throw new Error("seed 需为 -1 或 0..4294967295 的整数: " + String(raw));
  }
  if (!isFinite(n) || Math.floor(n) !== n || n < -1 || n > 4294967295) {
    throw new Error("seed 需为 -1 或 0..4294967295 的整数: " + String(raw));
  }
  return n === 0 ? 0 : n;
}

function stateIsFailed(st) {
  return st === "cancelled" || st === "canceled" || st === "error";
}

function collectImages(input) {
  if (isArray(input.images) && input.images.length > 0) {
    return input.images;
  }
  if (typeof input.image === "string" && input.image !== "") {
    return [input.image];
  }
  if (typeof input.first_frame === "string" && input.first_frame !== "") {
    if (typeof input.last_frame === "string" && input.last_frame !== "") {
      return [input.first_frame, input.last_frame];
    }
    return [input.first_frame];
  }
  return [];
}

function resolveVideoDimensions(aspectRatio, resolution, isRef2v) {
  var ar = String(aspectRatio || "Auto");
  var tier = String(resolution || "720P").toUpperCase();
  if (isRef2v && (ar === "16:9" || ar === "9:16" || ar === "Auto")) {
    if (tier !== "720P" && tier !== "1080P" && tier !== "4K") {
      throw new Error("不支持的 Ref2V 分辨率档位：" + tier);
    }
    var px = tier === "720P" ? [480, 864] : tier === "1080P" ? [544, 960] : [576, 1024];
    if (ar === "16:9") {
      return { width: px[1], height: px[0] };
    }
    return { width: px[0], height: px[1] };
  }
  var high = tier === "1080P" || tier === "4K";
  if (ar === "16:9") {
    return high ? { width: 1024, height: 576 } : { width: 864, height: 480 };
  }
  if (ar === "9:16" || ar === "Auto") {
    return high ? { width: 576, height: 1024 } : { width: 480, height: 864 };
  }
  if (ar === "4:3") {
    return { width: 768, height: 576 };
  }
  if (ar === "3:4") {
    return { width: 576, height: 768 };
  }
  if (ar === "1:1") {
    return high ? { width: 768, height: 768 } : { width: 640, height: 640 };
  }
  if (ar === "21:9") {
    return { width: 1024, height: 448 };
  }
  return { width: 864, height: 480 };
}

function resolveFrameLength(durationSec) {
  if (durationSec === undefined || durationSec === null || durationSec === "") {
    return 124;
  }
  var d = Number(durationSec);
  if (!isFinite(d) || Math.floor(d) !== d || d < 2 || d > 15) {
    throw new Error("duration 需为 2..15 的整数秒: " + durationSec);
  }
  var k = d < 5 ? Math.ceil((d * 24 - 22) / 17) : Math.round((d * 24 - 22) / 17);
  if (k < 0) k = 0;
  if (k > 20) k = 20;
  return 22 + k * 17;
}

function pickSetting(val, allowed, fallback) {
  var v = (val === undefined || val === null || val === "") ? fallback : String(val);
  if (allowed.indexOf(v) < 0) {
    throw new Error("不支持的渠道配置值：" + v);
  }
  return v;
}

function checkImages(arr, maxCount) {
  if (!isArray(arr) || arr.length === 0 || arr.length > maxCount) {
    throw new Error("参考图数量须在 1.." + maxCount + " 之间，当前为 " + (isArray(arr) ? arr.length : 0));
  }
  for (var i = 0; i < arr.length; i++) {
    if (!arr[i]) {
      throw new Error("第 " + (i + 1) + " 张参考图为空");
    }
  }
}

function normalizeRefPrompt(prompt, count) {
  var seen = {};
  var text = String(prompt || "");
  text = text.replace(/图片\s*([0-9]+(?:\.[0-9]+)?)/g, "<Picture $1>");
  text = text.replace(/<\s*picture\b([^>]*)>/gi, function (_, tail) {
    var v = String(tail).trim();
    if (!/^[1-9][0-9]*$/.test(v) || Number(v) > count) {
      throw new Error("Picture 引用越界或格式非法：<Picture " + v + ">（当前参考图 " + count + " 张）");
    }
    seen[v] = true;
    return "<Picture " + v + ">";
  });
  if (/<\s*picture/i.test(text.replace(/<Picture [1-9][0-9]*>/g, "")) || /图片\s*[-+.0-9０-９]/.test(text)) {
    throw new Error("参考图引用语法错误，请检查 <Picture N> 或 图片N");
  }
  if (count === 1 && !seen["1"]) {
    if (text.indexOf("[Shot 1]") >= 0) {
      return text.replace("[Shot 1]", "[Shot 1] <Picture 1>");
    }
    return "<Picture 1> " + text;
  }
  var missing = [];
  for (var i = 1; i <= count; i++) {
    if (!seen[String(i)]) {
      missing.push(String(i));
    }
  }
  if (missing.length > 0) {
    throw new Error("多图参考存在未在正文中引用的图片槽位：<Picture " + missing.join(">, <Picture ") + ">");
  }
  return text;
}

function isArray(val) {
  return Object.prototype.toString.call(val) === "[object Array]";
}
