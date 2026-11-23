using System.Net.Http;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;

namespace StoreGuard.Infrastructure.Engine;

public sealed class TheftEngineClient : IEngineClient, IApexSolutionClient
{
    private readonly string _baseHttpUrl;
    private readonly string _wsUrl;
    private readonly HttpClient _http;
    private CancellationTokenSource? _cts;
    private Task? _listenTask;

    public event EventHandler<TheftAlertEvent>? OnTheftAlert;
    public event EventHandler<CameraHealthEvent>? OnCameraHealth;
    public event EventHandler<VlmErrorEvent>? OnVlmError;
    public event EventHandler<EngineMetricsEvent>? OnEngineMetrics;
    public event EventHandler<bool>? OnConnectionChanged;

    public bool IsConnected { get; private set; }

    public TheftEngineClient(string baseUrl = "http://127.0.0.1:8700", HttpMessageHandler? handler = null)
    {
        _baseHttpUrl = baseUrl.TrimEnd('/');
        var uri = new Uri(_baseHttpUrl);
        var wsScheme = uri.Scheme.Equals("https", StringComparison.OrdinalIgnoreCase) ? "wss" : "ws";
        _wsUrl = $"{wsScheme}://{uri.Authority}/ws/events";
        _http = handler is null ? new HttpClient() : new HttpClient(handler);
        _http.Timeout = TimeSpan.FromSeconds(5);
    }

    public Task StartListeningAsync(CancellationToken ct)
    {
        if (_listenTask is not null)
            return Task.CompletedTask;
        _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _listenTask = Task.Run(() => ReceiveLoopAsync(_cts.Token), _cts.Token);
        return Task.CompletedTask;
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        var buffer = new byte[64 * 1024];
        while (!ct.IsCancellationRequested)
        {
            using var ws = new ClientWebSocket();
            try
            {
                await ws.ConnectAsync(new Uri(_wsUrl), ct);
                SetConnected(true);

                using var ms = new MemoryStream();
                while (ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
                {
                    ms.SetLength(0);
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await ws.ReceiveAsync(buffer, ct);
                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", ct);
                            break;
                        }
                        ms.Write(buffer, 0, result.Count);
                    } while (!result.EndOfMessage);

                    if (result.MessageType == WebSocketMessageType.Close)
                        break;

                    var json = Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
                    DispatchEnvelope(json);
                }
            }
            catch
            {
                // Auto-reconnect after backoff
            }

            SetConnected(false);
            if (!ct.IsCancellationRequested)
            {
                try { await Task.Delay(2000, ct); } catch { break; }
            }
        }
    }

    public void DispatchEnvelope(string rawJson)
    {
        using var doc = JsonDocument.Parse(rawJson);
        var root = doc.RootElement;
        if (!root.TryGetProperty("type", out var typeEl) || !root.TryGetProperty("payload", out var payloadEl))
            return;

        var msgType = typeEl.GetString() ?? string.Empty;
        var ts = root.TryGetProperty("timestamp_utc", out var tsEl) && DateTime.TryParse(tsEl.GetString(), out var parsedDt)
            ? parsedDt.ToUniversalTime()
            : DateTime.UtcNow;

        switch (msgType)
        {
            case "ALERT":
                var alert = JsonSerializer.Deserialize<TheftAlertEvent>(payloadEl.GetRawText());
                if (alert is not null)
                {
                    alert.TimestampUtc = ts;
                    OnTheftAlert?.Invoke(this, alert);
                }
                break;
            case "CAMERA_HEALTH":
                var health = JsonSerializer.Deserialize<CameraHealthEvent>(payloadEl.GetRawText());
                if (health is not null)
                    OnCameraHealth?.Invoke(this, health);
                break;
            case "VLM_ERROR":
                var vlmErr = JsonSerializer.Deserialize<VlmErrorEvent>(payloadEl.GetRawText());
                if (vlmErr is not null)
                    OnVlmError?.Invoke(this, vlmErr);
                break;
            case "ENGINE_METRICS":
                var metrics = JsonSerializer.Deserialize<EngineMetricsEvent>(payloadEl.GetRawText());
                if (metrics is not null)
                    OnEngineMetrics?.Invoke(this, metrics);
                break;
        }
    }

    private void SetConnected(bool value)
    {
        if (IsConnected == value) return;
        IsConnected = value;
        OnConnectionChanged?.Invoke(this, value);
    }

    public async Task<EngineHealthResponse?> GetHealthAsync(CancellationToken ct)
    {
        try
        {
            return await _http.GetFromJsonAsync<EngineHealthResponse>($"{_baseHttpUrl}/health", ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<byte[]?> GetSnapshotAsync(string cameraId, bool overlays, CancellationToken ct, int maxWidth = 0, int quality = 85, string? overlayLayers = null)
    {
        try
        {
            var ov = overlays ? "true" : "false";
            var url = $"{_baseHttpUrl}/cameras/{Uri.EscapeDataString(cameraId)}/snapshot?overlays={ov}";
            if (maxWidth > 0)
                url += $"&max_width={maxWidth}";
            if (quality != 85)
                url += $"&quality={Math.Clamp(quality, 40, 95)}";
            if (!string.IsNullOrWhiteSpace(overlayLayers))
                url += $"&layers={Uri.EscapeDataString(overlayLayers)}";
            using var resp = await _http.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadAsByteArrayAsync(ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<bool> UpdateCameraZonesAsync(string cameraId, IReadOnlyList<CameraZoneDto> zones, CancellationToken ct)
    {
        try
        {
            using var resp = await _http.PutAsJsonAsync($"{_baseHttpUrl}/cameras/{Uri.EscapeDataString(cameraId)}/zones", new { zones }, ct);
            return resp.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<bool> ReloadAllCamerasAsync(CancellationToken ct)
    {
        try
        {
            using var resp = await _http.PostAsync($"{_baseHttpUrl}/cameras/reload", null, ct);
            return resp.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<bool> RerunCameraDetectionAsync(string cameraId, CancellationToken ct)
    {
        try
        {
            using var resp = await _http.PostAsync($"{_baseHttpUrl}/cameras/{Uri.EscapeDataString(cameraId)}/rerun", null, ct);
            return resp.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<DeviceUpdateResponse?> SetDeviceAsync(string device, CancellationToken ct)
    {
        try
        {
            using var resp = await _http.PutAsJsonAsync($"{_baseHttpUrl}/engine/device", new { device }, ct);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadFromJsonAsync<DeviceUpdateResponse>(cancellationToken: ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<bool> RequestShutdownAsync(CancellationToken ct)
    {
        try
        {
            using var resp = await _http.PostAsync($"{_baseHttpUrl}/shutdown", null, ct);
            return resp.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    // --- IApexSolutionClient (Hikrobot VisionMaster-style Apex Solution & Procedure SDK) ---

    public async Task<ApexSolutionInfo?> GetSolutionAsync(CancellationToken ct = default)
    {
        try
        {
            return await _http.GetFromJsonAsync<ApexSolutionInfo>($"{_baseHttpUrl}/sdk/solution", ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<ApexSolutionXmlResponse?> GetSolutionXmlAsync(CancellationToken ct = default)
    {
        try
        {
            return await _http.GetFromJsonAsync<ApexSolutionXmlResponse>($"{_baseHttpUrl}/sdk/solution/xml", ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<ApexSolutionInfo?> LoadSolutionAsync(string? path = null, string? xml = null, CancellationToken ct = default)
    {
        try
        {
            var body = new ApexSolutionLoadRequest { Path = path, Xml = xml };
            using var resp = await _http.PostAsJsonAsync($"{_baseHttpUrl}/sdk/solution/load", body, ct);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadFromJsonAsync<ApexSolutionInfo>(cancellationToken: ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<ApexSolutionSaveResponse?> SaveSolutionAsync(string? path = null, CancellationToken ct = default)
    {
        try
        {
            using var resp = await _http.PostAsJsonAsync($"{_baseHttpUrl}/sdk/solution/save", new { path }, ct);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadFromJsonAsync<ApexSolutionSaveResponse>(cancellationToken: ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<List<ApexProcedureSummary>> ListProceduresAsync(CancellationToken ct = default)
    {
        try
        {
            using var resp = await _http.GetAsync($"{_baseHttpUrl}/sdk/procedures", ct);
            if (!resp.IsSuccessStatusCode) return new List<ApexProcedureSummary>();
            using var doc = await JsonDocument.ParseAsync(await resp.Content.ReadAsStreamAsync(ct), cancellationToken: ct);
            if (doc.RootElement.TryGetProperty("procedures", out var el))
                return JsonSerializer.Deserialize<List<ApexProcedureSummary>>(el.GetRawText()) ?? new List<ApexProcedureSummary>();
            return new List<ApexProcedureSummary>();
        }
        catch
        {
            return new List<ApexProcedureSummary>();
        }
    }

    public async Task<ApexProcedureSummary?> GetProcedureAsync(string name, CancellationToken ct = default)
    {
        try
        {
            return await _http.GetFromJsonAsync<ApexProcedureSummary>($"{_baseHttpUrl}/sdk/procedures/{Uri.EscapeDataString(name)}", ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<ApexProcedureRunResult?> RunProcedureOnceAsync(string name, CancellationToken ct = default)
    {
        try
        {
            using var resp = await _http.PostAsync($"{_baseHttpUrl}/sdk/procedures/{Uri.EscapeDataString(name)}/run", null, ct);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadFromJsonAsync<ApexProcedureRunResult>(cancellationToken: ct);
        }
        catch
        {
            return null;
        }
    }

    public async Task<ApexProcedureSummary?> SetProcedureParamAsync(
        string name,
        string nodeId,
        string paramName,
        object? value,
        CancellationToken ct = default)
    {
        try
        {
            var req = new ApexParamUpdateRequest { Value = value };
            using var resp = await _http.PutAsJsonAsync(
                $"{_baseHttpUrl}/sdk/procedures/{Uri.EscapeDataString(name)}/params/{Uri.EscapeDataString(nodeId)}/{Uri.EscapeDataString(paramName)}",
                req,
                ct);
            if (!resp.IsSuccessStatusCode) return null;
            return await resp.Content.ReadFromJsonAsync<ApexProcedureSummary>(cancellationToken: ct);
        }
        catch
        {
            return null;
        }
    }

    public async ValueTask DisposeAsync()
    {
        _cts?.Cancel();
        if (_listenTask is not null)
        {
            try { await _listenTask; } catch { }
        }
        _http.Dispose();
    }
}
