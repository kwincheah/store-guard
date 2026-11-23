using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;

namespace StoreGuard.Infrastructure.Notifications;

public sealed class TelegramNotifier : INotifier
{
    private readonly IEventRepository _repo;
    private readonly ISecretProtector _protector;
    private readonly HttpClient _http;

    public TelegramNotifier(IEventRepository repo, ISecretProtector protector, HttpClient? httpClient = null)
    {
        _repo = repo;
        _protector = protector;
        _http = httpClient ?? new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
    }

    public async Task<(bool Success, string Message)> SendTheftAlertAsync(TheftAlertEvent alert, CancellationToken ct)
    {
        var settings = await _repo.GetSettingsAsync(ct);
        if (!settings.TelegramEnabled)
            return (false, "Telegram disabled");

        var token = _protector.Unprotect(settings.EncryptedTelegramBotToken).Trim();
        var chatId = settings.TelegramChatId.Trim();
        if (string.IsNullOrEmpty(token) || string.IsNullOrEmpty(chatId))
            return (false, "Telegram token or chat ID missing");

        var caption =
            $"[StoreGuard Theft Alert]\n" +
            $"Camera: {alert.CameraName} ({alert.CameraId})\n" +
            $"Track #{alert.TrackId} | Confidence: {alert.ConfidencePct:F1}%\n" +
            $"Behaviors: {string.Join(", ", alert.Behaviors)}\n" +
            $"Time (UTC): {alert.TimestampUtc:yyyy-MM-dd HH:mm:ss}\n" +
            $"Summary: {alert.Justification}";

        return await SendPayloadAsync(token, chatId, caption, alert.GridPath, ct);
    }

    public Task<(bool Success, string Message)> SendTestMessageAsync(string botToken, string chatId, CancellationToken ct)
    {
        return SendPayloadAsync(
            botToken.Trim(),
            chatId.Trim(),
            "[StoreGuard] Test notification connected successfully.",
            null,
            ct);
    }

    private async Task<(bool Success, string Message)> SendPayloadAsync(
        string token,
        string chatId,
        string text,
        string? photoPath,
        CancellationToken ct)
    {
        if (string.IsNullOrEmpty(token) || string.IsNullOrEmpty(chatId))
            return (false, "Missing bot token or chat ID.");

        try
        {
            if (!string.IsNullOrWhiteSpace(photoPath) && File.Exists(photoPath))
            {
                using var form = new MultipartFormDataContent();
                form.Add(new StringContent(chatId), "chat_id");
                form.Add(new StringContent(text.Length > 1000 ? text[..1000] : text), "caption");
                var bytes = await File.ReadAllBytesAsync(photoPath, ct);
                var imgContent = new ByteArrayContent(bytes);
                imgContent.Headers.ContentType = MediaTypeHeaderValue.Parse("image/jpeg");
                form.Add(imgContent, "photo", Path.GetFileName(photoPath));

                using var resp = await _http.PostAsync($"https://api.telegram.org/bot{token}/sendPhoto", form, ct);
                if (!resp.IsSuccessStatusCode)
                    return (false, $"Telegram HTTP {(int)resp.StatusCode}");
                return (true, "Photo alert sent");
            }
            else
            {
                using var resp = await _http.PostAsJsonAsync(
                    $"https://api.telegram.org/bot{token}/sendMessage",
                    new { chat_id = chatId, text },
                    ct);
                if (!resp.IsSuccessStatusCode)
                    return (false, $"Telegram HTTP {(int)resp.StatusCode}");
                return (true, "Message sent");
            }
        }
        catch (Exception ex)
        {
            return (false, ex.Message);
        }
    }
}
