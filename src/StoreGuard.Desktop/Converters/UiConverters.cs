using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Data;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace StoreGuard.Desktop.Converters;

public sealed class BoolToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool flag = value is bool b && b;
        if (Invert) flag = !flag;
        return flag ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => value is Visibility v && v == Visibility.Visible;
}

public sealed class NullToVisibilityConverter : IValueConverter
{
    public bool Invert { get; set; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        bool notNull = value is not null;
        if (Invert) notNull = !notNull;
        return notNull ? Visibility.Visible : Visibility.Collapsed;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class StringNotEmptyToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        => string.IsNullOrWhiteSpace(value as string) ? Visibility.Collapsed : Visibility.Visible;

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class PathToBitmapConverter : IValueConverter
{
    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        var path = value as string;
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;
        try
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(path, UriKind.Absolute);
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
        catch
        {
            return null;
        }
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class BytesToBitmapConverter : IValueConverter
{
    public object? Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is not byte[] bytes || bytes.Length == 0)
            return null;
        try
        {
            using var ms = new MemoryStream(bytes);
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.StreamSource = ms;
            bmp.EndInit();
            bmp.Freeze();
            return bmp;
        }
        catch
        {
            return null;
        }
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class HealthStateToBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Online = Freeze(Color.FromRgb(16, 185, 129));
    private static readonly SolidColorBrush Reconnecting = Freeze(Color.FromRgb(245, 158, 11));
    private static readonly SolidColorBrush Offline = Freeze(Color.FromRgb(244, 63, 94));

    private static SolidColorBrush Freeze(Color c) { var b = new SolidColorBrush(c); b.Freeze(); return b; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        (value as string)?.ToUpperInvariant() switch
        {
            "ONLINE" => Online,
            "RECONNECTING" => Reconnecting,
            _ => Offline,
        };

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

internal static class ThemeResourceLookup
{
    public static SolidColorBrush Resolve(string key, SolidColorBrush fallback)
    {
        if (Application.Current?.TryFindResource(key) is SolidColorBrush brush)
            return brush;
        return fallback;
    }
}

public sealed class VerdictToBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Confirmed = Freeze(Color.FromRgb(190, 18, 60));
    private static readonly SolidColorBrush FalseAlarm = Freeze(Color.FromRgb(4, 120, 87));
    private static readonly SolidColorBrush Pending = Freeze(Color.FromRgb(217, 119, 6));

    private static SolidColorBrush Freeze(Color c) { var b = new SolidColorBrush(c); b.Freeze(); return b; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        (value as string) switch
        {
            "ConfirmedTheft" => ThemeResourceLookup.Resolve("DangerActionBrush", Confirmed),
            "FalseAlarm" => ThemeResourceLookup.Resolve("OkActionBrush", FalseAlarm),
            _ => ThemeResourceLookup.Resolve("WarnBrush", Pending),
        };

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class VerdictToTintBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Confirmed = Freeze(Color.FromRgb(53, 16, 26));
    private static readonly SolidColorBrush FalseAlarm = Freeze(Color.FromRgb(11, 46, 38));
    private static readonly SolidColorBrush Pending = Freeze(Color.FromRgb(50, 33, 8));

    private static SolidColorBrush Freeze(Color c) { var b = new SolidColorBrush(c); b.Freeze(); return b; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        (value as string) switch
        {
            "ConfirmedTheft" => ThemeResourceLookup.Resolve("DangerTintBrush", Confirmed),
            "FalseAlarm" => ThemeResourceLookup.Resolve("OkTintBrush", FalseAlarm),
            _ => ThemeResourceLookup.Resolve("WarnTintBrush", Pending),
        };

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class VerdictToTextBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Confirmed = Freeze(Color.FromRgb(253, 164, 175));
    private static readonly SolidColorBrush FalseAlarm = Freeze(Color.FromRgb(110, 231, 183));
    private static readonly SolidColorBrush Pending = Freeze(Color.FromRgb(252, 211, 77));

    private static SolidColorBrush Freeze(Color c) { var b = new SolidColorBrush(c); b.Freeze(); return b; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        (value as string) switch
        {
            "ConfirmedTheft" => ThemeResourceLookup.Resolve("DangerTextBrush", Confirmed),
            "FalseAlarm" => ThemeResourceLookup.Resolve("OkTextBrush", FalseAlarm),
            _ => ThemeResourceLookup.Resolve("WarnTextBrush", Pending),
        };

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class VerdictToBorderBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Confirmed = Freeze(Color.FromRgb(136, 19, 55));
    private static readonly SolidColorBrush FalseAlarm = Freeze(Color.FromRgb(17, 94, 73));
    private static readonly SolidColorBrush Pending = Freeze(Color.FromRgb(120, 53, 15));

    private static SolidColorBrush Freeze(Color c) { var b = new SolidColorBrush(c); b.Freeze(); return b; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        (value as string) switch
        {
            "ConfirmedTheft" => ThemeResourceLookup.Resolve("DangerBorderBrush", Confirmed),
            "FalseAlarm" => ThemeResourceLookup.Resolve("OkBorderBrush", FalseAlarm),
            _ => ThemeResourceLookup.Resolve("WarnBorderBrush", Pending),
        };

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}

public sealed class VerdictToAccentBrushConverter : IValueConverter
{
    private static readonly SolidColorBrush Confirmed = Freeze(Color.FromRgb(244, 63, 94));
    private static readonly SolidColorBrush FalseAlarm = Freeze(Color.FromRgb(16, 185, 129));
    private static readonly SolidColorBrush Pending = Freeze(Color.FromRgb(245, 158, 11));

    private static SolidColorBrush Freeze(Color c) { var b = new SolidColorBrush(c); b.Freeze(); return b; }

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture) =>
        (value as string) switch
        {
            "ConfirmedTheft" => ThemeResourceLookup.Resolve("DangerBrush", Confirmed),
            "FalseAlarm" => ThemeResourceLookup.Resolve("OkBrush", FalseAlarm),
            _ => ThemeResourceLookup.Resolve("WarnBrush", Pending),
        };

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        => throw new NotSupportedException();
}
