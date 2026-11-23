using System.IO;
using System.Text.Json;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using StoreGuard.Core.Models;

namespace StoreGuard.Desktop.ViewModels;

public sealed record FilterOption(string Value, string Label)
{
    public override string ToString() => Label;
}

public sealed partial class AlertItemViewModel : ObservableObject
{
    public string EventId { get; init; } = string.Empty;
    public string CameraId { get; init; } = string.Empty;
    public string CameraName { get; init; } = string.Empty;
    public int TrackId { get; init; }
    public string BehaviorCategory { get; init; } = string.Empty;
    public double ConfidencePct { get; init; }
    public string BehaviorsSummary { get; init; } = string.Empty;
    public string Justification { get; init; } = string.Empty;
    public string GridPath { get; init; } = string.Empty;
    public string ClipPath { get; init; } = string.Empty;
    public DateTime CreatedAtUtc { get; init; }

    public string LocalTimeDisplay => CreatedAtUtc.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss");
    public string ConfidenceDisplay => $"{ConfidencePct:F1}%";
    public bool HasValidClip => !string.IsNullOrWhiteSpace(ClipPath) && File.Exists(ClipPath);
    public string PlayClipTooltip => HasValidClip
        ? "Play 10s recorded evidence clip (Space)"
        : "10s MP4 clip unavailable or still encoding";

    [ObservableProperty] private string _verdict = "Unreviewed";
    [ObservableProperty] private string _reviewNote = string.Empty;
    [ObservableProperty] private string _reviewedBy = string.Empty;
    [ObservableProperty] private string _reviewedAtDisplay = string.Empty;

    partial void OnVerdictChanged(string value) => OnPropertyChanged(nameof(VerdictDisplay));
    partial void OnReviewNoteChanged(string value) => OnPropertyChanged(nameof(HasReviewNote));

    public bool HasReviewNote => !string.IsNullOrWhiteSpace(ReviewNote);

    public string VerdictDisplay => Verdict switch
    {
        "ConfirmedTheft" => "Confirmed Theft",
        "FalseAlarm" => "False Alarm",
        _ => "Unreviewed",
    };

    public string BehaviorsDisplay => FormatBehaviorsPretty(BehaviorsSummary);

    public static AlertItemViewModel FromRecord(EventRecord ev) => new()
    {
        EventId = ev.Id,
        CameraId = ev.CameraId,
        CameraName = ev.CameraName,
        TrackId = ev.TrackId,
        BehaviorCategory = ev.BehaviorCategory,
        ConfidencePct = ev.ConfidencePct,
        BehaviorsSummary = FormatBehaviorsJson(ev.BehaviorsJson),
        Justification = ev.Justification,
        GridPath = ev.GridPath,
        ClipPath = ev.ClipPath,
        CreatedAtUtc = ev.CreatedAtUtc,
        Verdict = ev.Review?.Verdict ?? "Unreviewed",
        ReviewNote = ev.Review?.Note ?? string.Empty,
        ReviewedBy = ev.Review?.ReviewedBy ?? string.Empty,
        ReviewedAtDisplay = ev.Review is null ? string.Empty : ev.Review.ReviewedAtUtc.ToLocalTime().ToString("yyyy-MM-dd HH:mm"),
    };

    private static string FormatBehaviorsPretty(string summary)
    {
        if (string.IsNullOrWhiteSpace(summary))
            return "Monitoring Sequence";

        var labels = new List<string>();
        foreach (var rawTag in summary.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var normalized = rawTag.ToLowerInvariant() switch
            {
                "reach_and_tuck" => "Reach & Tuck",
                "blading" => "Body Blading",
                "crouch" or "crouch_hide" => "Low Crouch",
                "bag_stuff" or "bag_concealment" => "Bag Concealment",
                "garment_hide" => "Garment Concealment",
                "jacket_pocket" => "Jacket Pocket",
                "shelf_item_removed" => "Shelf Item Removed",
                "carry_delayed" => "Delayed Carry",
                "shelf_linger" => "Shelf Linger",
                "reposition_item" => "Item Repositioned",
                _ => System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(rawTag.Replace('_', ' ')),
            };
            if (!labels.Contains(normalized))
                labels.Add(normalized);
        }
        return labels.Count > 0 ? string.Join(" · ", labels) : summary;
    }

    private static string FormatBehaviorsJson(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return string.Empty;
        try
        {
            var tags = JsonSerializer.Deserialize<List<string>>(raw);
            if (tags is { Count: > 0 })
                return string.Join(", ", tags);
        }
        catch
        {
            // Fall back to raw text if stored outside JSON array format
        }
        return raw;
    }
}

public sealed partial class CameraHealthPillViewModel : ObservableObject
{
    public string CameraId { get; init; } = string.Empty;
    [ObservableProperty] private string _name = string.Empty;
    [ObservableProperty] private string _state = "OFFLINE";
    [ObservableProperty] private double _fps;
    [ObservableProperty] private bool _motionActive;
    [ObservableProperty] private string _error = string.Empty;
}

public sealed partial class EditableZoneViewModel : ObservableObject
{
    [ObservableProperty] private string _name = "shelf_1";
    [ObservableProperty] private string _role = "shelf";
    [ObservableProperty] private double _xNorm;
    [ObservableProperty] private double _yNorm;
    [ObservableProperty] private double _wNorm;
    [ObservableProperty] private double _hNorm;

    partial void OnRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsShelf));
        OnPropertyChanged(nameof(RoleBadge));
    }

    partial void OnXNormChanged(double value) => OnPropertyChanged(nameof(CoordsSummary));
    partial void OnYNormChanged(double value) => OnPropertyChanged(nameof(CoordsSummary));
    partial void OnWNormChanged(double value) => OnPropertyChanged(nameof(CoordsSummary));
    partial void OnHNormChanged(double value) => OnPropertyChanged(nameof(CoordsSummary));

    public bool IsShelf => string.Equals(Role, "shelf", StringComparison.OrdinalIgnoreCase);
    public string RoleBadge => IsShelf ? "SHELF" : "AISLE";
    public string CoordsSummary => $"x:{XNorm:F2}  y:{YNorm:F2}  ·  {WNorm:F2} × {HNorm:F2}";
    public string Summary => $"{Name} [{Role}] ({XNorm:F2}, {YNorm:F2}, {WNorm:F2}x{HNorm:F2})";
}

public sealed partial class CameraSnapshotTileViewModel : ObservableObject
{
    public string CameraId { get; init; } = string.Empty;
    [ObservableProperty] private string _name = string.Empty;
    [ObservableProperty] private string _state = "OFFLINE";
    [ObservableProperty] private double _fps;
    [ObservableProperty] private byte[]? _snapshotBytes;
    [ObservableProperty] private ImageSource? _snapshotImage;
    [ObservableProperty] private bool _isSelected = true;
    [ObservableProperty] private bool _isLocalFile;
    [ObservableProperty] private bool _clipDetectionDone;
    [ObservableProperty] private bool _isRerunning;
    [ObservableProperty] private double _feedAspectRatio = 16.0 / 9.0;
    [ObservableProperty] private double _tileMediaWidth = 480.0;
    [ObservableProperty] private double _tileMediaHeight = 270.0;
    [ObservableProperty] private double _tileMaxWidth = 500.0;

    partial void OnStateChanged(string value)
    {
        OnPropertyChanged(nameof(IsOffline));
        OnPropertyChanged(nameof(StateBadge));
        OnPropertyChanged(nameof(OfflineMessage));
    }

    partial void OnFpsChanged(double value) => OnPropertyChanged(nameof(StateBadge));

    partial void OnSnapshotBytesChanged(byte[]? value) => OnPropertyChanged(nameof(HasSnapshot));

    partial void OnSnapshotImageChanged(ImageSource? value) => OnPropertyChanged(nameof(HasSnapshot));

    partial void OnIsLocalFileChanged(bool value)
    {
        OnPropertyChanged(nameof(ShowRerunButton));
        OnPropertyChanged(nameof(HasDetectionModeBadge));
        OnPropertyChanged(nameof(DetectionModeBadge));
    }

    partial void OnClipDetectionDoneChanged(bool value) => OnPropertyChanged(nameof(DetectionModeBadge));

    partial void OnIsRerunningChanged(bool value) => OnPropertyChanged(nameof(DetectionModeBadge));

    public bool IsOffline => !string.Equals(State, "ONLINE", StringComparison.OrdinalIgnoreCase);
    public bool HasSnapshot => SnapshotImage is not null || SnapshotBytes is { Length: > 0 };
    public bool ShowRerunButton => IsLocalFile;
    public bool HasDetectionModeBadge => IsLocalFile;
    public string DetectionModeBadge => !IsLocalFile
        ? string.Empty
        : (IsRerunning ? "Restarting..." : (ClipDetectionDone ? "Preview · Done" : "Detecting"));
    public string StateBadge => IsOffline ? (string.Equals(State, "RECONNECTING", StringComparison.OrdinalIgnoreCase) ? "RECONNECTING" : "OFFLINE") : $"{Fps:F1} fps";
    public string OfflineMessage => string.Equals(State, "RECONNECTING", StringComparison.OrdinalIgnoreCase)
        ? "RECONNECTING — Waiting for stream..."
        : "OFFLINE — Camera stream unavailable";

    public bool UpdateFeedDimensions(int pixelWidth, int pixelHeight)
    {
        if (pixelWidth <= 0 || pixelHeight <= 0)
            return false;
        double clampedAr = Math.Clamp(pixelWidth / (double)pixelHeight, 0.45, 2.60);
        if (Math.Abs(clampedAr - FeedAspectRatio) <= 0.02)
            return false;
        FeedAspectRatio = clampedAr;
        return true;
    }
}
