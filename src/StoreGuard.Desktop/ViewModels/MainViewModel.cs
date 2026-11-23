using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Desktop.Services;
using StoreGuard.Infrastructure.Engine;
using StoreGuard.Infrastructure.Sdk;
using StoreGuard.Infrastructure.Services;

namespace StoreGuard.Desktop.ViewModels;

public sealed partial class MainViewModel : ObservableObject
{
    private readonly IEventRepository _repo;
    private readonly ISecretProtector _protector;
    private readonly IEngineClient _client;
    private readonly INotifier _notifier;
    private readonly CamerasJsonWriter _configWriter;
    private readonly PythonProcessHost _processHost;
    private readonly SoundAlertService _sound;
    private readonly string _configPath;
    private readonly AppFileLogger? _logger;

    public AlertFeedViewModel AlertFeed { get; }
    public LiveGridViewModel LiveGrid { get; }
    public HistoryViewModel History { get; }
    public CameraSetupViewModel CameraSetup { get; }
    public SettingsViewModel Settings { get; }
    public ApexFlowEditorViewModel ApexFlowEditor { get; }

    [ObservableProperty] private int _selectedTabIndex;
    [ObservableProperty] private bool _isNavAlerts = true;
    [ObservableProperty] private bool _isNavLiveGrid;
    [ObservableProperty] private bool _isNavHistory;
    [ObservableProperty] private bool _isNavOwner;
    [ObservableProperty] private string _activePageTitle = "Counter Alerts";
    [ObservableProperty] private string _activePageSubtitle = "Real-time AI theft triage & counter review";

    [ObservableProperty] private bool _isOwnerUnlocked;
    [ObservableProperty] private bool _isDefaultPinActive = true;
    [ObservableProperty] private bool _isEngineConnected;
    [ObservableProperty] private bool _isSoundMuted;
    [ObservableProperty] private bool _isLightTheme;
    [ObservableProperty] private int _unreviewedCount;
    [ObservableProperty] private string _engineBanner = "Starting AI engine...";
    [ObservableProperty] private string _vlmWarningBanner = string.Empty;
    [ObservableProperty] private string _gpuMetricsText = "Device: initializing...";

    [ObservableProperty] private string _selectedDevice = "auto";
    [ObservableProperty] private bool _isDeviceAuto = true;
    [ObservableProperty] private bool _isDeviceCpu;
    [ObservableProperty] private bool _isDeviceGpu;
    [ObservableProperty] private bool _isUsingGpu;
    [ObservableProperty] private string _activeHardwareBadge = "CPU";
    [ObservableProperty] private string _deviceFallbackHint = string.Empty;

    public event Func<Task<string?>>? RequestOwnerPinAsync;
    public event Func<Func<string, Task<bool>>, bool, Task<bool>>? RequestOwnerUnlockAsync;
    public event Action<int>? RequestAdminSubTab;

    public MainViewModel(
        IEventRepository repo,
        ISecretProtector protector,
        IEngineClient client,
        INotifier notifier,
        CamerasJsonWriter configWriter,
        PythonProcessHost processHost,
        SoundAlertService sound,
        string configPath,
        AppFileLogger? logger = null,
        Func<string, string, string, bool, bool>? confirmDialog = null,
        ApexSolution? apexSolution = null,
        string? solutionPath = null)
    {
        _repo = repo;
        _protector = protector;
        _client = client;
        _notifier = notifier;
        _configWriter = configWriter;
        _processHost = processHost;
        _sound = sound;
        _configPath = configPath;
        _logger = logger;

        AlertFeed = new AlertFeedViewModel(repo);
        AlertFeed.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(AlertFeedViewModel.UnreviewedCount))
                UnreviewedCount = AlertFeed.UnreviewedCount;
        };
        LiveGrid = new LiveGridViewModel(client, repo);
        LiveGrid.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(LiveGridViewModel.ShowRoiOverlays) && SelectedTabIndex == 1)
                ActivePageSubtitle = BuildLiveGridSubtitle();
        };
        History = new HistoryViewModel(repo);
        History.ReviewUpdated += AlertFeed.SyncReviewFromHistory;
        CameraSetup = new CameraSetupViewModel(repo, protector, client, SyncConfigAndReloadEngineAsync, WriteCamerasJsonAsync, logger, confirmDialog);
        Settings = new SettingsViewModel(repo, protector, notifier, SyncConfigAndReloadEngineAsync, logger);
        Settings.OwnerPinUpdated += (_, _) => _ = RefreshDefaultPinStateAsync();
        Settings.ThemeRequested += (_, theme) => _ = ApplyAndPersistThemeAsync(theme);
        var resolvedSolution = apexSolution ?? (client as IApexSolutionClient is { } solClient ? new ApexSolution(solClient) : null);
        ApexFlowEditor = new ApexFlowEditorViewModel(
            () => IsOwnerUnlocked,
            ApplyApexEditorZonesToCameraSetup,
            confirmDialog: confirmDialog,
            onSolutionSaved: async path =>
            {
                if (resolvedSolution is null)
                    return false;
                var reloaded = await resolvedSolution.LoadAsync(path);
                return reloaded is not null;
            },
            activeSolutionPath: solutionPath ?? processHost.SolutionPath);

        _processHost.OnStatusMessage += msg =>
        {
            _logger?.Info($"EngineHost: {msg}");
            RunOnUi(() => EngineBanner = msg);
        };
        _client.OnConnectionChanged += (_, connected) =>
        {
            _logger?.Info($"Engine WebSocket connected={connected}");
            RunOnUi(() =>
            {
                IsEngineConnected = connected;
                EngineBanner = connected ? "Engine Online (:8700)" : "Reconnecting Engine...";
            });
        };
        _client.OnTheftAlert += (_, alert) => _ = HandleIncomingAlertAsync(alert);
        _client.OnCameraHealth += (_, health) => RunOnUi(() =>
        {
            AlertFeed.UpdateCameraHealth(health);
            LiveGrid.ApplyHealthEvent(health);
        });
        _client.OnVlmError += (_, err) =>
        {
            _logger?.Warn($"VLM_ERROR cam={err.CameraId} track=#{err.TrackId}: {err.Error}");
            RunOnUi(() =>
                VlmWarningBanner = $"VLM Error on {err.CameraId} (Track #{err.TrackId}): {err.Error}");
        };
        _client.OnEngineMetrics += (_, m) => RunOnUi(() => ApplyHardwareTelemetry(m.RequestedDevice, m.Device, m.UsingGpu, m.ActiveStreams, m.VlmQueueDepth, m.FallbackNote));
    }

    public async Task InitializeAsync(CancellationToken ct)
    {
        await _repo.InitializeAsync(ct);
        var s = await _repo.GetSettingsAsync(ct);
        IsSoundMuted = s.SoundMuted;
        ApplyDeviceButtons(s.Device);
        var normalizedTheme = ThemeManager.NormalizeTheme(s.Theme);
        IsLightTheme = normalizedTheme == "light";
        RunOnUi(() => ThemeManager.ApplyTheme(normalizedTheme));
        IsDefaultPinActive = await _repo.IsUsingDefaultPinAsync(ct);

        await AlertFeed.LoadInitialAsync();
        await CameraSetup.LoadAsync();
        await Settings.LoadAsync();
        await History.RefreshAsync();

        await WriteCamerasJsonAsync();
        await _processHost.StartAndMonitorAsync(ct);
        bool ready = await _processHost.WaitForHealthyAsync(TimeSpan.FromSeconds(25), ct);
        if (!ready)
            EngineBanner = "AI Engine starting slowly — check runtime_data/logs/ai_engine.log";

        await _client.StartListeningAsync(ct);
        var health = await _client.GetHealthAsync(ct);
        if (health is not null)
        {
            RunOnUi(() => ApplyHardwareTelemetry(health.RequestedDevice, health.Device, health.UsingGpu, health.ActiveStreams, health.VlmQueueDepth, health.FallbackNote));
        }
    }

    private string BuildLiveGridSubtitle()
        => LiveGrid.ShowRoiOverlays
            ? "Live camera feeds with person boxes, Track IDs & ROI overlays"
            : "Live camera feeds (overlays hidden)";

    private async Task RefreshDefaultPinStateAsync()
    {
        try
        {
            IsDefaultPinActive = await _repo.IsUsingDefaultPinAsync();
        }
        catch
        {
        }
    }

    partial void OnSelectedTabIndexChanged(int value)
    {
        IsNavAlerts = value == 0;
        IsNavLiveGrid = value == 1;
        IsNavHistory = value == 2;
        IsNavOwner = value == 3;
        (ActivePageTitle, ActivePageSubtitle) = value switch
        {
            1 => ("Live Camera Grid", BuildLiveGridSubtitle()),
            2 => ("History & Audit", "Filter incidents, review evidence, and export CSV"),
            3 => ("Admin Settings", "Per-camera RTSP setup, ROI zones, compute device, VLM & alerts"),
            _ => ("Counter Alerts", "Real-time AI theft triage & counter review"),
        };

        if (value == 1)
            _ = LiveGrid.StartAsync();
        else
            LiveGrid.Stop();

        if (value == 2)
            _ = History.RefreshAsync();
    }

    [RelayCommand]
    private async Task ToggleThemeAsync()
    {
        var next = IsLightTheme ? "dark" : "light";
        await ApplyAndPersistThemeAsync(next);
    }

    public async Task ApplyAndPersistThemeAsync(string? theme)
    {
        var normalized = ThemeManager.NormalizeTheme(theme);
        IsLightTheme = normalized == "light";
        Settings.Theme = normalized;
        RunOnUi(() => ThemeManager.ApplyTheme(normalized));

        try
        {
            var s = await _repo.GetSettingsAsync();
            if (!string.Equals(s.Theme, normalized, StringComparison.OrdinalIgnoreCase))
            {
                s.Theme = normalized;
                await _repo.SaveSettingsAsync(s);
            }
        }
        catch
        {
            // Keep UI theme responsive even if DB is read-only/unavailable
        }
    }

    [RelayCommand]
    private async Task SelectNavTabAsync(string indexText)
    {
        if (!int.TryParse(indexText, out int idx)) return;
        if (idx == 3 && !IsOwnerUnlocked)
        {
            await UnlockOwnerAsync();
            if (!IsOwnerUnlocked)
            {
                IsNavAlerts = SelectedTabIndex == 0;
                IsNavLiveGrid = SelectedTabIndex == 1;
                IsNavHistory = SelectedTabIndex == 2;
                IsNavOwner = false;
            }
            return;
        }
        SelectedTabIndex = idx;
    }

    [RelayCommand]
    private async Task OpenCameraSetupAsync()
    {
        await UnlockOwnerAsync();
        if (IsOwnerUnlocked)
        {
            SelectedTabIndex = 3;
            RequestAdminSubTab?.Invoke(0);
        }
    }

    [RelayCommand]
    private async Task SelectDeviceAsync(string device)
    {
        var req = string.IsNullOrWhiteSpace(device) ? "auto" : device.Trim().ToLowerInvariant();
        ApplyDeviceButtons(req);
        Settings.Device = req;

        var s = await _repo.GetSettingsAsync();
        s.Device = req;
        await _repo.SaveSettingsAsync(s);
        await WriteCamerasJsonAsync();

        var res = await _client.SetDeviceAsync(req, CancellationToken.None);
        if (res is not null)
        {
            RunOnUi(() => ApplyHardwareTelemetry(res.RequestedDevice, res.ResolvedDevice, res.UsingGpu, AlertFeed.CameraHealths.Count, 0, res.FallbackNote));
        }
    }

    private void ApplyDeviceButtons(string requestedDevice)
    {
        var req = (requestedDevice ?? "auto").Trim().ToLowerInvariant();
        SelectedDevice = req;
        IsDeviceAuto = req == "auto";
        IsDeviceCpu = req == "cpu";
        IsDeviceGpu = req.StartsWith("cuda");
    }

    private void ApplyHardwareTelemetry(string requested, string resolved, bool usingGpu, int activeStreams, int vlmQueue, string fallbackNote)
    {
        ApplyDeviceButtons(requested);
        IsUsingGpu = usingGpu;
        ActiveHardwareBadge = usingGpu ? $"GPU ({resolved.ToUpperInvariant()})" : "CPU";
        DeviceFallbackHint = fallbackNote ?? string.Empty;
        Settings.ActiveHardwareNote = string.IsNullOrWhiteSpace(fallbackNote)
            ? $"Active hardware: {ActiveHardwareBadge}"
            : $"{ActiveHardwareBadge} — {fallbackNote}";
        GpuMetricsText = $"{ActiveHardwareBadge}\nStreams: {activeStreams} · VLM Queue: {vlmQueue}";
    }

    partial void OnIsSoundMutedChanged(bool value)
    {
        _ = SaveMutePreferenceAsync(value);
    }

    private async Task SaveMutePreferenceAsync(bool muted)
    {
        var s = await _repo.GetSettingsAsync();
        s.SoundMuted = muted;
        await _repo.SaveSettingsAsync(s);
    }

    [RelayCommand]
    private async Task UnlockOwnerAsync()
    {
        if (IsOwnerUnlocked)
        {
            SelectedTabIndex = 3;
            return;
        }

        if (RequestOwnerUnlockAsync is not null)
        {
            bool unlocked = await RequestOwnerUnlockAsync(pin => _repo.VerifyOwnerPinAsync(pin), IsDefaultPinActive);
            if (!unlocked) return;

            IsOwnerUnlocked = true;
            _logger?.Info("Admin settings unlocked.");
            ApexFlowEditor.TryLoadDefault();
            SelectedTabIndex = 3;
            return;
        }

        if (RequestOwnerPinAsync is null) return;
        var pin = await RequestOwnerPinAsync();
        if (string.IsNullOrEmpty(pin)) return;

        if (await _repo.VerifyOwnerPinAsync(pin))
        {
            IsOwnerUnlocked = true;
            _logger?.Info("Admin settings unlocked.");
            ApexFlowEditor.TryLoadDefault();
            SelectedTabIndex = 3;
        }
        else
        {
            _logger?.Warn("Failed Admin PIN verification attempt.");
        }
    }

    [RelayCommand]
    private void LockOwner()
    {
        IsOwnerUnlocked = false;
        _logger?.Info("Admin settings locked.");
        if (SelectedTabIndex == 3)
            SelectedTabIndex = 0;
    }

    [RelayCommand]
    private void DismissVlmWarning() => VlmWarningBanner = string.Empty;

    private async Task HandleIncomingAlertAsync(TheftAlertEvent alert)
    {
        _logger?.Info($"Incoming ALERT id={alert.EventId} cam={alert.CameraId} track=#{alert.TrackId} conf={alert.ConfidencePct:F1}%");
        try
        {
            await _repo.InsertEventAsync(alert);
        }
        catch (Exception ex)
        {
            _logger?.Error($"Failed inserting alert {alert.EventId} to SQLite", ex);
        }

        RunOnUi(() =>
        {
            AlertFeed.AddIncomingAlert(alert);
            _sound.PlayAlert(IsSoundMuted);
            _ = History.RefreshAsync();
        });

        var (ok, msg) = await _notifier.SendTheftAlertAsync(alert, CancellationToken.None);
        _logger?.Info($"Telegram alert dispatch for {alert.EventId}: ok={ok}, msg={msg}");
        if (msg.StartsWith("Telegram HTTP", StringComparison.Ordinal))
        {
            RunOnUi(() => VlmWarningBanner = $"Telegram push failed: {msg}");
        }
    }

    private async Task WriteCamerasJsonAsync()
    {
        var s = await _repo.GetSettingsAsync();
        var cams = await _repo.GetCamerasAsync();
        _configWriter.WriteConfigAtomic(_configPath, s, cams);
    }

    private async Task SyncConfigAndReloadEngineAsync()
    {
        await WriteCamerasJsonAsync();
        await _client.ReloadAllCamerasAsync(CancellationToken.None);
        var health = await _client.GetHealthAsync(CancellationToken.None);
        RunOnUi(() =>
        {
            _ = AlertFeed.LoadInitialAsync();
            if (health is not null)
                ApplyHardwareTelemetry(health.RequestedDevice, health.Device, health.UsingGpu, health.ActiveStreams, health.VlmQueueDepth, health.FallbackNote);
        });
    }

    public async Task ShutdownAsync()
    {
        LiveGrid.Stop();
        await _client.DisposeAsync();
        await _processHost.DisposeAsync();
    }

    private void ApplyApexEditorZonesToCameraSetup(string streamUrl, string solutionName, List<CameraZone> zones)
    {
        if (CameraSetup.SelectedCamera is null && CameraSetup.Cameras.Count > 0)
            CameraSetup.SelectedCamera = CameraSetup.Cameras[0];
        if (string.IsNullOrWhiteSpace(CameraSetup.EditCameraId))
            CameraSetup.NewBlankCameraCommand.Execute(null);
        CameraSetup.ApplyParsedApex(streamUrl, solutionName, zones, "Apex Flow Editor");
    }

    private static void RunOnUi(Action action)
    {
        var disp = Application.Current?.Dispatcher;
        if (disp is null || disp.CheckAccess())
            action();
        else
            disp.BeginInvoke(action);
    }
}
