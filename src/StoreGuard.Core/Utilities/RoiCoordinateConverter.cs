using StoreGuard.Core.Models;

namespace StoreGuard.Core.Utilities;

public static class RoiCoordinateConverter
{
    public static double Clamp01(double value) => Math.Clamp(value, 0.0, 1.0);

    public static double[] PixelsToNormalized(double x, double y, double w, double h, double imageWidth, double imageHeight)
    {
        if (imageWidth <= 0 || imageHeight <= 0 || w <= 0 || h <= 0)
            return [0, 0, 0, 0];

        var nx = Clamp01(x / imageWidth);
        var ny = Clamp01(y / imageHeight);
        var nw = Clamp01(w / imageWidth);
        var nh = Clamp01(h / imageHeight);

        if (nx + nw > 1.0) nw = Math.Max(0.0, 1.0 - nx);
        if (ny + nh > 1.0) nh = Math.Max(0.0, 1.0 - ny);

        return
        [
            Math.Round(nx, 6),
            Math.Round(ny, 6),
            Math.Round(nw, 6),
            Math.Round(nh, 6),
        ];
    }

    public static (double X, double Y, double W, double H) NormalizedToPixels(CameraZone zone, double displayWidth, double displayHeight)
    {
        if (displayWidth <= 0 || displayHeight <= 0)
            return (0, 0, 0, 0);

        return (
            Clamp01(zone.XNorm) * displayWidth,
            Clamp01(zone.YNorm) * displayHeight,
            Clamp01(zone.WNorm) * displayWidth,
            Clamp01(zone.HNorm) * displayHeight
        );
    }

    public static CameraZoneDto ToDto(CameraZone zone) => new()
    {
        Name = zone.Name,
        Role = zone.Role,
        RectNorm = [Math.Round(Clamp01(zone.XNorm), 6), Math.Round(Clamp01(zone.YNorm), 6), Math.Round(Clamp01(zone.WNorm), 6), Math.Round(Clamp01(zone.HNorm), 6)],
    };
}
