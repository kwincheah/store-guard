using System.Net.Http;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Infrastructure.Services;

namespace StoreGuard.Desktop.ViewModels;

public sealed record DeviceOption(string Value, string Label)
{
    public override string ToString() => Label;
}

public sealed partial class SettingsViewModel : ObservableObject
{
    private static readonly IReadOnlyDictionary<string, string> ProviderDefaultModels =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["gemini"] = "gemini-2.5-flash",
            ["openai"] = "gpt-4o-mini",
            ["deepseek"] = "deepseek-chat",
            ["anthropic"] = "claude-sonnet-4-20250514",
            ["openrouter"] = "openai/gpt-4o-mini",
            ["ollama"] = "llama3.2-vision",
            ["lmstudio"] = "local-model",
            ["custom"] = "custom-model",
            ["dry_run"] = "dry-run",
        };

    private static readonly HashSet<string> KnownDefaultModelNames =
        new(ProviderDefaultModels.Values, StringComparer.OrdinalIgnoreCase);

    private readonly IEventRepository _repo;
    private readonly ISecretProtector _protector;
    private readonly INotifier _notifier;
    private readonly Func<Task> _syncAndReloadEngineAsync;
    private readonly AppFileLogger? _logger;
    private readonly Func<HttpMessageHandler?>? _vlmProbeHandlerFactory;
    private bool _isLoading;

    public event EventHandler? OwnerPinUpdated;
    public event EventHandler<string>? ThemeRequested;

    public IReadOnlyList<DeviceOption> DeviceChoices { get; } =
    [
        new("auto", "Auto (Prefer GPU)"),
        new("cpu", "CPU Only"),
        new("cuda:0", "NVIDIA GPU (CUDA 0)"),
    ];
    public IReadOnlyList<string> ProviderChoices { get; } = new[] { "gemini", "openai", "deepseek", "anthropic", "openrouter", "ollama", "lmstudio", "custom", "dry_run" };
    public IReadOnlyList<string> PoseModelChoices { get; } = new[] { "yolo11n-pose.onnx", "yolo11s-pose.onnx", "yolo11m-pose.onnx" };
    public IReadOnlyList<int> BagStrideChoices { get; } = new[] { 1, 2, 3, 4, 5, 6 };

    [ObservableProperty] private string _device = "auto";
    [ObservableProperty] private string _poseModel = "yolo11n-pose.onnx";
    [ObservableProperty] private int _bagInferStride = 3;
    [ObservableProperty] private string _activeHardwareNote = "Resolving device...";
    [ObservableProperty] private int _targetFps = 8;
    [ObservableProperty] private int _retentionDays = 30;
    [ObservableProperty] private double _maxEvidenceGb = 20.0;
    [ObservableProperty] private string _vlmProvider = "gemini";
    [ObservableProperty] private string _vlmBaseUrl = string.Empty;
    [ObservableProperty] private string _vlmModelName = "gemini-2.5-flash";
    [ObservableProperty] private string _vlmApiKey = string.Empty;
    [ObservableProperty] private double _alertThresholdPct = 65.0;
    [ObservableProperty] private bool _vlmUseNativeVideo = true;
    [ObservableProperty] private string _telegramBotToken = string.Empty;
    [ObservableProperty] private string _telegramChatId = string.Empty;
    [ObservableProperty] private bool _telegramEnabled;
    [ObservableProperty] private string _newOwnerPin = string.Empty;
    [ObservableProperty] private string _statusMessage = string.Empty;
    [ObservableProperty] private string _theme = "dark";
    [ObservableProperty] private bool _hasUnsavedChanges;

    public bool IsGeminiProvider => string.Equals(VlmProvider, "gemini", StringComparison.OrdinalIgnoreCase);

    public SettingsViewModel(
        IEventRepository repo,
        ISecretProtector protector,
        INotifier notifier,
        Func<Task> syncAndReloadEngineAsync,
        AppFileLogger? logger = null,
        Func<HttpMessageHandler?>? vlmProbeHandlerFactory = null)
    {
        _repo = repo;
        _protector = protector;
        _notifier = notifier;
        _syncAndReloadEngineAsync = syncAndReloadEngineAsync;
        _logger = logger;
        _vlmProbeHandlerFactory = vlmProbeHandlerFactory;
    }

    partial void OnDeviceChanged(string value) => MarkDirty();
    partial void OnPoseModelChanged(string value) => MarkDirty();
    partial void OnBagInferStrideChanged(int value) => MarkDirty();
    partial void OnTargetFpsChanged(int value) => MarkDirty();
    partial void OnRetentionDaysChanged(int value) => MarkDirty();
    partial void OnMaxEvidenceGbChanged(double value) => MarkDirty();
    partial void OnVlmBaseUrlChanged(string value) => MarkDirty();
    partial void OnVlmModelNameChanged(string value) => MarkDirty();
    partial void OnVlmApiKeyChanged(string value) => MarkDirty();
    partial void OnAlertThresholdPctChanged(double value) => MarkDirty();
    partial void OnVlmUseNativeVideoChanged(bool value) => MarkDirty();
    partial void OnTelegramBotTokenChanged(string value) => MarkDirty();
    partial void OnTelegramChatIdChanged(string value) => MarkDirty();
    partial void OnTelegramEnabledChanged(bool value) => MarkDirty();

    partial void OnVlmProviderChanged(string value)
    {
        OnPropertyChanged(nameof(IsGeminiProvider));
        if (!_isLoading
            && ProviderDefaultModels.TryGetValue((value ?? string.Empty).Trim(), out var suggested)
            && (string.IsNullOrWhiteSpace(VlmModelName) || KnownDefaultModelNames.Contains(VlmModelName.Trim())))
        {
            VlmModelName = suggested;
        }
        MarkDirty();
    }

    partial void OnThemeChanged(string value)
    {
        if (!_isLoading)
            ThemeRequested?.Invoke(this, value);
    }

    private void MarkDirty()
    {
        if (!_isLoading)
            HasUnsavedChanges = true;
    }

    public async Task LoadAsync()
    {
        _isLoading = true;
        try
        {
            var s = await _repo.GetSettingsAsync();
            Device = s.Device;
            PoseModel = string.IsNullOrWhiteSpace(s.PoseModel) ? "yolo11n-pose.onnx" : s.PoseModel;
            BagInferStride = Math.Clamp(s.BagInferStride, 1, 6);
            TargetFps = s.TargetFps;
            RetentionDays = s.RetentionDays;
            MaxEvidenceGb = s.MaxEvidenceGb;
            VlmProvider = s.VlmProvider;
            VlmBaseUrl = s.VlmBaseUrl;
            VlmModelName = s.VlmModelName;
            VlmApiKey = _protector.Unprotect(s.EncryptedVlmApiKey);
            AlertThresholdPct = s.AlertThresholdPct;
            VlmUseNativeVideo = s.VlmUseNativeVideo;
            TelegramBotToken = _protector.Unprotect(s.EncryptedTelegramBotToken);
            TelegramChatId = s.TelegramChatId;
            TelegramEnabled = s.TelegramEnabled;
            Theme = string.Equals(s.Theme, "light", StringComparison.OrdinalIgnoreCase) ? "light" : "dark";
            HasUnsavedChanges = false;
        }
        finally
        {
            _isLoading = false;
        }
    }

    [RelayCommand]
    private async Task DiscardSettingsAsync()
    {
        await LoadAsync();
        StatusMessage = "Reverted unsaved settings changes.";
    }

    [RelayCommand]
    private async Task SaveSettingsAsync()
    {
        var existing = await _repo.GetSettingsAsync();
        var model = new AppSettingsModel
        {
            Id = 1,
            Device = Device,
            PoseModel = string.IsNullOrWhiteSpace(PoseModel) ? "yolo11n-pose.onnx" : PoseModel.Trim(),
            BagInferStride = Math.Clamp(BagInferStride, 1, 6),
            TargetFps = Math.Clamp(TargetFps, 1, 30),
            ClipPreSec = existing.ClipPreSec,
            ClipPostSec = existing.ClipPostSec,
            RetentionDays = Math.Max(1, RetentionDays),
            MaxEvidenceGb = Math.Max(0.5, MaxEvidenceGb),
            VlmProvider = VlmProvider,
            VlmBaseUrl = VlmBaseUrl.Trim(),
            VlmModelName = VlmModelName.Trim(),
            EncryptedVlmApiKey = _protector.Protect(VlmApiKey.Trim()),
            AlertThresholdPct = Math.Clamp(AlertThresholdPct, 10.0, 99.0),
            VlmUseNativeVideo = VlmUseNativeVideo,
            EncryptedTelegramBotToken = _protector.Protect(TelegramBotToken.Trim()),
            TelegramChatId = TelegramChatId.Trim(),
            TelegramEnabled = TelegramEnabled,
            SoundMuted = existing.SoundMuted,
            Theme = string.Equals(Theme, "light", StringComparison.OrdinalIgnoreCase) ? "light" : "dark",
        };
        await _repo.SaveSettingsAsync(model);
        await _syncAndReloadEngineAsync();
        HasUnsavedChanges = false;
        _logger?.Info($"Settings saved: device={model.Device}, pose={model.PoseModel}, stride={model.BagInferStride}, fps={model.TargetFps}, provider={model.VlmProvider}, model={model.VlmModelName}, native_video={model.VlmUseNativeVideo}, threshold={model.AlertThresholdPct}, telegram={model.TelegramEnabled}, theme={model.Theme}");
        StatusMessage = "Settings saved and pushed to AI engine.";
    }

    [RelayCommand]
    private async Task TestVlmConnectionAsync()
    {
        var provider = (VlmProvider ?? string.Empty).Trim().ToLowerInvariant();
        if (provider == "dry_run")
        {
            StatusMessage = "VLM Test OK: dry_run active (local simulation without API calls).";
            return;
        }

        var key = (VlmApiKey ?? string.Empty).Trim();
        var baseUrl = (VlmBaseUrl ?? string.Empty).Trim().TrimEnd('/');
        if (provider is "gemini" or "openai" or "deepseek" or "anthropic" or "openrouter" && string.IsNullOrWhiteSpace(key))
        {
            StatusMessage = $"VLM Test blocked: enter an API key for '{provider}' first.";
            return;
        }

        string probeUrl = provider switch
        {
            "gemini" => string.IsNullOrWhiteSpace(baseUrl)
                ? "https://generativelanguage.googleapis.com/v1beta/models"
                : $"{baseUrl}/models",
            "openai" => string.IsNullOrWhiteSpace(baseUrl)
                ? "https://api.openai.com/v1/models"
                : $"{baseUrl}/models",
            "deepseek" => string.IsNullOrWhiteSpace(baseUrl)
                ? "https://api.deepseek.com/models"
                : $"{baseUrl}/models",
            "anthropic" => string.IsNullOrWhiteSpace(baseUrl)
                ? "https://api.anthropic.com/v1/models"
                : $"{baseUrl}/v1/models",
            "openrouter" => string.IsNullOrWhiteSpace(baseUrl)
                ? "https://openrouter.ai/api/v1/models"
                : $"{baseUrl}/models",
            "ollama" => string.IsNullOrWhiteSpace(baseUrl)
                ? "http://127.0.0.1:11434/api/tags"
                : $"{baseUrl}/api/tags",
            "lmstudio" => string.IsNullOrWhiteSpace(baseUrl)
                ? "http://127.0.0.1:1234/v1/models"
                : $"{baseUrl}/models",
            _ => string.IsNullOrWhiteSpace(baseUrl) ? string.Empty : $"{baseUrl}/models",
        };

        if (string.IsNullOrWhiteSpace(probeUrl))
        {
            StatusMessage = "VLM Test blocked: provide a Custom Base URL for custom provider.";
            return;
        }

        StatusMessage = $"Testing {provider} connection...";
        try
        {
            var customHandler = _vlmProbeHandlerFactory?.Invoke();
            using var http = customHandler is not null
                ? new HttpClient(customHandler, disposeHandler: false) { Timeout = TimeSpan.FromSeconds(8) }
                : new HttpClient { Timeout = TimeSpan.FromSeconds(8) };

            using var req = new HttpRequestMessage(HttpMethod.Get, probeUrl);
            if (provider == "gemini" && !string.IsNullOrWhiteSpace(key))
                req.Headers.Add("x-goog-api-key", key);
            else if (provider == "anthropic" && !string.IsNullOrWhiteSpace(key))
            {
                req.Headers.Add("x-api-key", key);
                req.Headers.Add("anthropic-version", "2023-06-01");
            }
            else if (!string.IsNullOrWhiteSpace(key))
            {
                req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", key);
            }

            using var resp = await http.SendAsync(req, CancellationToken.None);
            _logger?.Info($"VLM connection probe for provider={provider}: HTTP {(int)resp.StatusCode}");
            StatusMessage = resp.IsSuccessStatusCode
                ? $"VLM Test OK ({provider}): HTTP {(int)resp.StatusCode} reachable."
                : $"VLM Test Failed ({provider}): HTTP {(int)resp.StatusCode} ({resp.ReasonPhrase}).";
        }
        catch (Exception ex)
        {
            _logger?.Warn($"VLM connection probe failed for provider={provider}: {ex.Message}");
            StatusMessage = $"VLM Test Error ({provider}): {ex.Message}";
        }
    }

    [RelayCommand]
    private async Task SendTestTelegramAsync()
    {
        var (ok, msg) = await _notifier.SendTestMessageAsync(TelegramBotToken, TelegramChatId, CancellationToken.None);
        _logger?.Info($"Telegram test notification: ok={ok}, msg={msg}");
        StatusMessage = ok ? $"Telegram OK: {msg}" : $"Telegram Failed: {msg}";
    }

    [RelayCommand]
    private async Task ChangeOwnerPinAsync()
    {
        var pin = NewOwnerPin.Trim();
        if (pin.Length < 4)
        {
            StatusMessage = "Admin PIN must be at least 4 characters.";
            return;
        }
        await _repo.UpdateOwnerPinAsync(pin);
        NewOwnerPin = string.Empty;
        _logger?.Info("Admin PIN updated.");
        StatusMessage = "Admin PIN updated.";
        OwnerPinUpdated?.Invoke(this, EventArgs.Empty);
    }
}
