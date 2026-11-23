using System.Media;

namespace StoreGuard.Infrastructure.Services;

public sealed class SoundAlertService
{
    private readonly string _wavPath;

    public SoundAlertService(string? wavPath = null)
    {
        _wavPath = wavPath ?? Path.Combine(AppContext.BaseDirectory, "Assets", "alert_chime.wav");
    }

    public void PlayAlert(bool muted)
    {
        if (muted || !File.Exists(_wavPath))
            return;
        try
        {
            using var player = new SoundPlayer(_wavPath);
            player.Play();
        }
        catch
        {
            // Ignore audio hardware failures on headless test environments
        }
    }
}
