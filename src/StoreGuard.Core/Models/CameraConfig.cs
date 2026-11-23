using System.Text.Json.Serialization;

namespace StoreGuard.Core.Models;

public sealed class CameraZone
{
    public int Id { get; set; }
    public string CameraId { get; set; } = string.Empty;
    public string Name { get; set; } = "zone";
    public string Role { get; set; } = "shelf"; // "shelf" | "aisle"
    public double XNorm { get; set; }
    public double YNorm { get; set; }
    public double WNorm { get; set; }
    public double HNorm { get; set; }
}

public sealed class CameraZoneDto
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "zone";

    [JsonPropertyName("role")]
    public string Role { get; set; } = "shelf";

    [JsonPropertyName("rect_norm")]
    public double[] RectNorm { get; set; } = [0, 0, 0, 0];
}

public sealed class CameraConfig
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string EncryptedRtspUrl { get; set; } = string.Empty;
    public string EncryptedSubStreamUrl { get; set; } = string.Empty;
    public bool Enabled { get; set; } = true;
    public double DwellSec { get; set; } = 3.0;
    public double ReleaseWaitSec { get; set; } = 3.5;
    public double CooldownSec { get; set; } = 12.0;
    public List<CameraZone> Zones { get; set; } = new();
}
