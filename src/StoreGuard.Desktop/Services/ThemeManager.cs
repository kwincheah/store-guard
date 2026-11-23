using System.Windows;

namespace StoreGuard.Desktop.Services;

public static class ThemeManager
{
    private const string DarkUri = "pack://application:,,,/StoreGuard.Desktop;component/Styles/DesignTokens.xaml";
    private const string LightUri = "pack://application:,,,/StoreGuard.Desktop;component/Styles/LightTheme.xaml";

    public static string CurrentTheme { get; private set; } = "dark";
    public static bool IsLightTheme => string.Equals(CurrentTheme, "light", StringComparison.OrdinalIgnoreCase);

    public static event EventHandler<string>? ThemeChanged;

    public static string NormalizeTheme(string? theme)
        => string.Equals(theme?.Trim(), "light", StringComparison.OrdinalIgnoreCase) ? "light" : "dark";

    public static void ApplyTheme(string? theme)
    {
        var normalized = NormalizeTheme(theme);
        CurrentTheme = normalized;

        var app = Application.Current;
        if (app is null)
        {
            ThemeChanged?.Invoke(null, normalized);
            return;
        }

        var merged = app.Resources.MergedDictionaries;
        int tokenIdx = -1;
        for (int i = 0; i < merged.Count; i++)
        {
            var src = merged[i].Source?.ToString() ?? string.Empty;
            if (src.Contains("DesignTokens.xaml", StringComparison.OrdinalIgnoreCase)
                || src.Contains("LightTheme.xaml", StringComparison.OrdinalIgnoreCase))
            {
                tokenIdx = i;
                break;
            }
        }

        var nextDict = new ResourceDictionary
        {
            Source = new Uri(normalized == "light" ? LightUri : DarkUri, UriKind.Absolute),
        };

        if (tokenIdx >= 0)
            merged[tokenIdx] = nextDict;
        else
            merged.Insert(0, nextDict);

        ThemeChanged?.Invoke(null, normalized);
    }
}
