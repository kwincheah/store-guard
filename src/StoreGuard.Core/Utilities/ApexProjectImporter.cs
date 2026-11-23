using System.Text.Json;
using System.Xml.Linq;
using StoreGuard.Core.Models;

namespace StoreGuard.Core.Utilities;

public static class ApexProjectImporter
{
    public static (string StreamUrl, string SolutionName, List<CameraZone> Zones) ParseApexFile(
        string apexFilePath,
        string cameraId,
        int refWidth = 1280,
        int refHeight = 720)
    {
        var doc = XDocument.Load(apexFilePath);
        return ParseDocument(doc, cameraId, refWidth, refHeight);
    }

    public static (string StreamUrl, string SolutionName, List<CameraZone> Zones) ParseApexXml(
        string apexXmlContent,
        string cameraId,
        int refWidth = 1280,
        int refHeight = 720)
    {
        var doc = XDocument.Parse(apexXmlContent);
        return ParseDocument(doc, cameraId, refWidth, refHeight);
    }

    private static (string StreamUrl, string SolutionName, List<CameraZone> Zones) ParseDocument(
        XDocument doc,
        string cameraId,
        int refWidth,
        int refHeight)
    {
        var root = doc.Root ?? throw new InvalidDataException("Invalid .apex XML file.");
        var solutionName = (string?)root.Attribute("name") ?? cameraId;
        var streamUrl = string.Empty;
        var zones = new List<CameraZone>();

        foreach (var node in root.Descendants("Node"))
        {
            var toolName = ((string?)node.Attribute("tool_name") ?? string.Empty).Trim();
            var parameters = node.Element("Parameters")?.Elements("Parameter") ?? Enumerable.Empty<XElement>();

            if (toolName == "Image Source" && string.IsNullOrEmpty(streamUrl))
            {
                var p = parameters.FirstOrDefault(x => (string?)x.Attribute("name") == "stream_url");
                streamUrl = (p?.Value ?? string.Empty).Trim();
            }
            else if (toolName == "ROI Zone Manager")
            {
                var zp = parameters.FirstOrDefault(x => (string?)x.Attribute("name") == "zones");
                if (zp is null || string.IsNullOrWhiteSpace(zp.Value))
                    continue;

                using var jsonDoc = JsonDocument.Parse(zp.Value);
                if (jsonDoc.RootElement.ValueKind != JsonValueKind.Array)
                    continue;

                int idx = 0;
                foreach (var item in jsonDoc.RootElement.EnumerateArray())
                {
                    var name = item.TryGetProperty("name", out var nEl) ? (nEl.GetString() ?? $"zone_{idx}") : $"zone_{idx}";
                    idx++;
                    if (!item.TryGetProperty("rect", out var rEl) || rEl.ValueKind != JsonValueKind.Array || rEl.GetArrayLength() < 4)
                        continue;

                    double x = rEl[0].GetDouble();
                    double y = rEl[1].GetDouble();
                    double w = rEl[2].GetDouble();
                    double h = rEl[3].GetDouble();
                    if (w <= 0 || h <= 0)
                        continue;

                    var norm = RoiCoordinateConverter.PixelsToNormalized(x, y, w, h, Math.Max(1, refWidth), Math.Max(1, refHeight));
                    zones.Add(new CameraZone
                    {
                        CameraId = cameraId,
                        Name = name,
                        Role = name.Contains("shelf", StringComparison.OrdinalIgnoreCase) ? "shelf" : "aisle",
                        XNorm = norm[0],
                        YNorm = norm[1],
                        WNorm = norm[2],
                        HNorm = norm[3],
                    });
                }
            }
        }

        return (streamUrl, solutionName, zones);
    }
}
