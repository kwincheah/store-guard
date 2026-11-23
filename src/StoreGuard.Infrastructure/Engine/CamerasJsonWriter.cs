using System.Text.Json;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Core.Utilities;

namespace StoreGuard.Infrastructure.Engine;

public sealed class CamerasJsonWriter
{
    private readonly ISecretProtector _protector;

    public CamerasJsonWriter(ISecretProtector protector)
    {
        _protector = protector;
    }

    public void WriteConfigAtomic(string targetPath, AppSettingsModel settings, IEnumerable<CameraConfig> cameras)
    {
        var dir = Path.GetDirectoryName(targetPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        var payload = new
        {
            engine = new
            {
                device = settings.Device,
                target_fps = settings.TargetFps,
                clip_pre_sec = settings.ClipPreSec,
                clip_post_sec = settings.ClipPostSec,
                retention_days = settings.RetentionDays,
                max_evidence_gb = settings.MaxEvidenceGb,
                evidence_dir = "runtime_data/evidence",
                models_dir = "models",
                pose_conf = 0.25,
                pose_model = string.IsNullOrWhiteSpace(settings.PoseModel) ? "yolo11n-pose.onnx" : settings.PoseModel.Trim(),
                bag_model = "yolo11n.onnx",
                bag_infer_stride = Math.Clamp(settings.BagInferStride, 1, 6),
                vlm = new
                {
                    provider = settings.VlmProvider,
                    base_url = settings.VlmBaseUrl,
                    model_name = settings.VlmModelName,
                    api_key = _protector.Unprotect(settings.EncryptedVlmApiKey),
                    timeout_sec = 25.0,
                    alert_threshold_pct = settings.AlertThresholdPct,
                    use_native_video = settings.VlmUseNativeVideo,
                    video_fps = 2.0,
                    max_sequence_frames = 14,
                },
            },
            cameras = cameras.Select(c => new
            {
                camera_id = c.Id,
                name = c.Name,
                rtsp_url = _protector.Unprotect(c.EncryptedRtspUrl),
                sub_stream_url = _protector.Unprotect(c.EncryptedSubStreamUrl),
                enabled = c.Enabled,
                dwell_sec = c.DwellSec,
                release_wait_sec = c.ReleaseWaitSec,
                cooldown_sec = c.CooldownSec,
                zones = c.Zones.Select(RoiCoordinateConverter.ToDto).ToList(),
            }).ToList(),
        };

        var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
        var tmp = targetPath + ".tmp";
        File.WriteAllText(tmp, json);
        File.Move(tmp, targetPath, overwrite: true);
    }
}
