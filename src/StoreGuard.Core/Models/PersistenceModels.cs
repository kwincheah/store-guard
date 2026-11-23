namespace StoreGuard.Core.Models;

public sealed class EventRecord
{
    public string Id { get; set; } = string.Empty;
    public string CameraId { get; set; } = string.Empty;
    public string CameraName { get; set; } = string.Empty;
    public int TrackId { get; set; }
    public string BehaviorCategory { get; set; } = string.Empty;
    public double ConfidencePct { get; set; }
    public string BehaviorsJson { get; set; } = "[]";
    public string Justification { get; set; } = string.Empty;
    public string GridPath { get; set; } = string.Empty;
    public string ClipPath { get; set; } = string.Empty;
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
    public EventReview? Review { get; set; }
}

public sealed class EventReview
{
    public int Id { get; set; }
    public string EventId { get; set; } = string.Empty;
    public int? UserId { get; set; }
    public string ReviewedBy { get; set; } = "CounterStaff";
    public string Verdict { get; set; } = string.Empty; // ConfirmedTheft | FalseAlarm
    public string Note { get; set; } = string.Empty;
    public DateTime ReviewedAtUtc { get; set; } = DateTime.UtcNow;
}

public sealed class UserAccount
{
    public int Id { get; set; }
    public string Username { get; set; } = "admin";
    public string DisplayName { get; set; } = "Store Admin";
    public string Role { get; set; } = "Admin";
    public string PinHash { get; set; } = string.Empty;
    public bool IsActive { get; set; } = true;
}

public sealed class AppSettingsModel
{
    public int Id { get; set; } = 1;
    public string Device { get; set; } = "auto";
    public string PoseModel { get; set; } = "yolo11n-pose.onnx";
    public int TargetFps { get; set; } = 8;
    public double ClipPreSec { get; set; } = 6.0;
    public double ClipPostSec { get; set; } = 4.0;
    public int RetentionDays { get; set; } = 30;
    public double MaxEvidenceGb { get; set; } = 20.0;
    public string VlmProvider { get; set; } = "gemini";
    public string VlmBaseUrl { get; set; } = string.Empty;
    public string VlmModelName { get; set; } = "gemini-2.5-flash";
    public string EncryptedVlmApiKey { get; set; } = string.Empty;
    public double AlertThresholdPct { get; set; } = 65.0;
    public bool VlmUseNativeVideo { get; set; } = true;
    public string EncryptedTelegramBotToken { get; set; } = string.Empty;
    public string TelegramChatId { get; set; } = string.Empty;
    public bool TelegramEnabled { get; set; }
    public bool SoundMuted { get; set; }
    public string Theme { get; set; } = "dark";
    public int BagInferStride { get; set; } = 3;
}
