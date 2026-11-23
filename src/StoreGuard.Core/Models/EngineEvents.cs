using System.Text.Json.Serialization;

namespace StoreGuard.Core.Models;

public sealed class TheftAlertEvent
{
    [JsonPropertyName("event_id")]
    public string EventId { get; set; } = string.Empty;

    [JsonPropertyName("camera_id")]
    public string CameraId { get; set; } = string.Empty;

    [JsonPropertyName("camera_name")]
    public string CameraName { get; set; } = string.Empty;

    [JsonPropertyName("track_id")]
    public int TrackId { get; set; }

    [JsonPropertyName("behavior_category")]
    public string BehaviorCategory { get; set; } = string.Empty;

    [JsonPropertyName("confidence_pct")]
    public double ConfidencePct { get; set; }

    [JsonPropertyName("behaviors")]
    public List<string> Behaviors { get; set; } = new();

    [JsonPropertyName("nearby_bags")]
    public List<string> NearbyBags { get; set; } = new();

    [JsonPropertyName("justification")]
    public string Justification { get; set; } = string.Empty;

    [JsonPropertyName("grid_path")]
    public string GridPath { get; set; } = string.Empty;

    [JsonPropertyName("clip_path")]
    public string ClipPath { get; set; } = string.Empty;

    [JsonIgnore]
    public DateTime TimestampUtc { get; set; } = DateTime.UtcNow;
}

public sealed class CameraHealthEvent
{
    [JsonPropertyName("camera_id")]
    public string CameraId { get; set; } = string.Empty;

    [JsonPropertyName("state")]
    public string State { get; set; } = "OFFLINE"; // ONLINE | RECONNECTING | OFFLINE

    [JsonPropertyName("fps")]
    public double Fps { get; set; }

    [JsonPropertyName("motion_active")]
    public bool MotionActive { get; set; }

    [JsonPropertyName("last_frame_age_ms")]
    public int LastFrameAgeMs { get; set; }

    [JsonPropertyName("error")]
    public string Error { get; set; } = string.Empty;

    [JsonPropertyName("is_local_file")]
    public bool IsLocalFile { get; set; }

    [JsonPropertyName("clip_detection_done")]
    public bool ClipDetectionDone { get; set; }

    [JsonPropertyName("loop_count")]
    public int LoopCount { get; set; }

    [JsonPropertyName("decoder_backend")]
    public string DecoderBackend { get; set; } = string.Empty;
}

public sealed class VlmErrorEvent
{
    [JsonPropertyName("camera_id")]
    public string CameraId { get; set; } = string.Empty;

    [JsonPropertyName("track_id")]
    public int TrackId { get; set; }

    [JsonPropertyName("error")]
    public string Error { get; set; } = string.Empty;

    [JsonPropertyName("grid_path")]
    public string GridPath { get; set; } = string.Empty;
}

public sealed class EngineMetricsEvent
{
    [JsonPropertyName("device")]
    public string Device { get; set; } = "cpu";

    [JsonPropertyName("requested_device")]
    public string RequestedDevice { get; set; } = "auto";

    [JsonPropertyName("using_gpu")]
    public bool UsingGpu { get; set; }

    [JsonPropertyName("cuda_available")]
    public bool CudaAvailable { get; set; }

    [JsonPropertyName("fallback_note")]
    public string FallbackNote { get; set; } = string.Empty;

    [JsonPropertyName("active_streams")]
    public int ActiveStreams { get; set; }

    [JsonPropertyName("worker_threads")]
    public int WorkerThreads { get; set; }

    [JsonPropertyName("tensorrt_available")]
    public bool TensorrtAvailable { get; set; }

    [JsonPropertyName("vlm_queue_depth")]
    public int VlmQueueDepth { get; set; }
}

public sealed class DeviceUpdateResponse
{
    [JsonPropertyName("requested_device")]
    public string RequestedDevice { get; set; } = "auto";

    [JsonPropertyName("resolved_device")]
    public string ResolvedDevice { get; set; } = "cpu";

    [JsonPropertyName("using_gpu")]
    public bool UsingGpu { get; set; }

    [JsonPropertyName("cuda_available")]
    public bool CudaAvailable { get; set; }

    [JsonPropertyName("fallback_note")]
    public string FallbackNote { get; set; } = string.Empty;
}

public sealed class EngineHealthResponse
{
    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("runtime")]
    public string Runtime { get; set; } = "apex_vision";

    [JsonPropertyName("solution_name")]
    public string SolutionName { get; set; } = string.Empty;

    [JsonPropertyName("solution_path")]
    public string SolutionPath { get; set; } = string.Empty;

    [JsonPropertyName("device")]
    public string Device { get; set; } = "cpu";

    [JsonPropertyName("requested_device")]
    public string RequestedDevice { get; set; } = "auto";

    [JsonPropertyName("using_gpu")]
    public bool UsingGpu { get; set; }

    [JsonPropertyName("cuda_available")]
    public bool CudaAvailable { get; set; }

    [JsonPropertyName("fallback_note")]
    public string FallbackNote { get; set; } = string.Empty;

    [JsonPropertyName("active_streams")]
    public int ActiveStreams { get; set; }

    [JsonPropertyName("worker_threads")]
    public int WorkerThreads { get; set; }

    [JsonPropertyName("tensorrt_available")]
    public bool TensorrtAvailable { get; set; }

    [JsonPropertyName("vlm_queue_depth")]
    public int VlmQueueDepth { get; set; }

    [JsonPropertyName("cameras")]
    public List<CameraHealthEvent> Cameras { get; set; } = new();
}
