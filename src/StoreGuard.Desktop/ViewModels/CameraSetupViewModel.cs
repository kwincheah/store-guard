using System.Collections.ObjectModel;
using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Core.Utilities;
using StoreGuard.Infrastructure.Services;

namespace StoreGuard.Desktop.ViewModels;

public sealed partial class CameraSetupViewModel : ObservableObject
{
    private readonly IEventRepository _repo;
    private readonly ISecretProtector _protector;
    private readonly IEngineClient _client;
    private readonly Func<Task> _syncAndReloadEngineAsync;
    private readonly Func<Task>? _writeCamerasJsonAsync;
    private readonly AppFileLogger? _logger;
    private readonly Func<string, string, string, bool, bool>? _confirmDialog;
    private readonly Stack<List<CameraZone>> _zoneUndoStack = new();

    private CancellationTokenSource? _snapshotCts;
    private bool _preserveSnapshotOnNextSelect;
    private bool _isLoadingSelection;
    private bool _isRevertingSelection;
    private CameraConfig? _committedSelection;

    public ObservableCollection<CameraConfig> Cameras { get; } = new();
    public ObservableCollection<EditableZoneViewModel> Zones { get; } = new();
    public IReadOnlyList<string> ZoneRoles { get; } = new[] { "shelf", "aisle" };

    [ObservableProperty] private CameraConfig? _selectedCamera;
    [ObservableProperty] private EditableZoneViewModel? _selectedZone;
    [ObservableProperty] private string _editCameraId = string.Empty;
    [ObservableProperty] private string _editName = string.Empty;
    [ObservableProperty] private string _editRtspUrl = string.Empty;
    [ObservableProperty] private bool _editEnabled = true;
    [ObservableProperty] private bool _hasUnsavedChanges;

    partial void OnEditCameraIdChanged(string value) => MarkDirtyIfInteractive();
    partial void OnEditNameChanged(string value) => MarkDirtyIfInteractive();
    partial void OnEditEnabledChanged(bool value) => MarkDirtyIfInteractive();
    partial void OnEditDwellSecChanged(double value) => MarkDirtyIfInteractive();
    partial void OnEditReleaseWaitSecChanged(double value) => MarkDirtyIfInteractive();
    partial void OnEditCooldownSecChanged(double value) => MarkDirtyIfInteractive();

    partial void OnEditRtspUrlChanged(string value)
    {
        OnPropertyChanged(nameof(IsLocalVideoSource));
        OnPropertyChanged(nameof(IsRtspSource));
        OnPropertyChanged(nameof(SourceKindBadge));
        MarkDirtyIfInteractive();
    }

    public bool IsLocalVideoSource
    {
        get
        {
            var raw = (EditRtspUrl ?? string.Empty).Trim();
            if (string.IsNullOrEmpty(raw)) return false;
            return raw.StartsWith("file://", StringComparison.OrdinalIgnoreCase)
                || !raw.Contains("://", StringComparison.Ordinal);
        }
    }

    public bool IsRtspSource => !string.IsNullOrWhiteSpace(EditRtspUrl) && !IsLocalVideoSource;

    public string SourceKindBadge
    {
        get
        {
            if (string.IsNullOrWhiteSpace(EditRtspUrl)) return "NO SOURCE";
            return IsLocalVideoSource ? "LOCAL VIDEO LOOP" : "RTSP STREAM";
        }
    }

    [ObservableProperty] private double _editDwellSec = 3.0;
    [ObservableProperty] private double _editReleaseWaitSec = 3.5;
    [ObservableProperty] private double _editCooldownSec = 12.0;

    [ObservableProperty] private string _selectedDrawRole = "shelf";
    [ObservableProperty] private string _newZoneName = "shelf_1";

    public bool IsDrawShelf => !string.Equals(SelectedDrawRole, "aisle", StringComparison.OrdinalIgnoreCase);
    public bool IsDrawAisle => string.Equals(SelectedDrawRole, "aisle", StringComparison.OrdinalIgnoreCase);
    public bool CanUndoZoneEdit => _zoneUndoStack.Count > 0;

    partial void OnSelectedDrawRoleChanged(string value)
    {
        OnPropertyChanged(nameof(IsDrawShelf));
        OnPropertyChanged(nameof(IsDrawAisle));
        var role = IsDrawAisle ? "aisle" : "shelf";
        var cur = (NewZoneName ?? string.Empty).Trim();
        if (string.IsNullOrEmpty(cur)
            || cur.StartsWith("shelf_", StringComparison.OrdinalIgnoreCase)
            || cur.StartsWith("aisle_", StringComparison.OrdinalIgnoreCase))
        {
            int nextIdx = Zones.Count(z => string.Equals(z.Role, role, StringComparison.OrdinalIgnoreCase)) + 1;
            NewZoneName = $"{role}_{nextIdx}";
        }
    }

    [RelayCommand]
    private void SelectDrawRole(string role)
    {
        SelectedDrawRole = string.Equals(role, "aisle", StringComparison.OrdinalIgnoreCase) ? "aisle" : "shelf";
    }

    [ObservableProperty] private byte[]? _snapshotBytes;
    [ObservableProperty] private string _statusMessage = string.Empty;

    // Uniview Quick-Add fields
    [ObservableProperty] private string _unvIp = "192.168.1.50";
    [ObservableProperty] private int _unvPort = 554;
    [ObservableProperty] private string _unvUser = "admin";
    [ObservableProperty] private string _unvPassword = string.Empty;
    [ObservableProperty] private int _unvChannelStart = 1;
    [ObservableProperty] private int _unvChannelEnd = 10;

    public CameraSetupViewModel(
        IEventRepository repo,
        ISecretProtector protector,
        IEngineClient client,
        Func<Task> syncAndReloadEngineAsync,
        Func<Task>? writeCamerasJsonAsync = null,
        AppFileLogger? logger = null,
        Func<string, string, string, bool, bool>? confirmDialog = null)
    {
        _repo = repo;
        _protector = protector;
        _client = client;
        _syncAndReloadEngineAsync = syncAndReloadEngineAsync;
        _writeCamerasJsonAsync = writeCamerasJsonAsync;
        _logger = logger;
        _confirmDialog = confirmDialog;
    }

    private string DisplayCameraLabel
        => string.IsNullOrWhiteSpace(EditName)
            ? (string.IsNullOrWhiteSpace(EditCameraId) ? "camera" : EditCameraId.Trim())
            : EditName.Trim();

    private void MarkDirtyIfInteractive()
    {
        if (!_isLoadingSelection)
            HasUnsavedChanges = true;
    }

    private void AttachZone(EditableZoneViewModel z) => z.PropertyChanged += OnZoneItemPropertyChanged;
    private void DetachAllZones()
    {
        foreach (var z in Zones)
            z.PropertyChanged -= OnZoneItemPropertyChanged;
    }

    private void OnZoneItemPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName is nameof(EditableZoneViewModel.Name)
            or nameof(EditableZoneViewModel.Role)
            or nameof(EditableZoneViewModel.XNorm)
            or nameof(EditableZoneViewModel.YNorm)
            or nameof(EditableZoneViewModel.WNorm)
            or nameof(EditableZoneViewModel.HNorm))
        {
            MarkDirtyIfInteractive();
        }
    }

    public void PushZoneUndoSnapshot()
    {
        var snap = Zones.Select(z => new CameraZone
        {
            CameraId = EditCameraId,
            Name = z.Name,
            Role = z.Role,
            XNorm = z.XNorm,
            YNorm = z.YNorm,
            WNorm = z.WNorm,
            HNorm = z.HNorm,
        }).ToList();
        _zoneUndoStack.Push(snap);
        OnPropertyChanged(nameof(CanUndoZoneEdit));
    }

    private void ClearUndoStack()
    {
        _zoneUndoStack.Clear();
        OnPropertyChanged(nameof(CanUndoZoneEdit));
    }

    public async Task LoadAsync(string? preserveCameraId = null, bool preserveSnapshot = false)
    {
        var savedSnapshot = preserveSnapshot ? SnapshotBytes : null;
        _preserveSnapshotOnNextSelect = preserveSnapshot;
        _isLoadingSelection = true;
        try
        {
            var list = await _repo.GetCamerasAsync();
            Cameras.Clear();
            foreach (var c in list) Cameras.Add(c);

            var match = !string.IsNullOrWhiteSpace(preserveCameraId)
                ? Cameras.FirstOrDefault(c => string.Equals(c.Id, preserveCameraId.Trim(), StringComparison.OrdinalIgnoreCase))
                : null;
            var next = match ?? Cameras.FirstOrDefault();
            if (match is null && preserveSnapshot)
                SnapshotBytes = null;

            HasUnsavedChanges = false;
            ApplyCameraSelection(next);
            if (preserveSnapshot && match is not null && SnapshotBytes is null && savedSnapshot is not null)
                SnapshotBytes = savedSnapshot;
        }
        finally
        {
            _preserveSnapshotOnNextSelect = false;
            _isLoadingSelection = false;
        }
    }

    partial void OnSelectedCameraChanged(CameraConfig? value)
    {
        if (_isRevertingSelection) return;

        if (!_isLoadingSelection && HasUnsavedChanges && _confirmDialog is not null && _committedSelection is not null && !ReferenceEquals(value, _committedSelection))
        {
            var currentLabel = string.IsNullOrWhiteSpace(EditName) ? _committedSelection.Name : EditName;
            bool discard = _confirmDialog(
                "Discard Unsaved Changes",
                $"'{currentLabel}' has unsaved camera or ROI edits. Switch away and discard them?",
                "Discard Edits",
                true);
            if (!discard)
            {
                _isRevertingSelection = true;
                try
                {
                    SelectedCamera = _committedSelection;
                }
                finally
                {
                    _isRevertingSelection = false;
                }
                return;
            }
        }

        ApplyCameraSelection(value);
    }

    private void ApplyCameraSelection(CameraConfig? value)
    {
        _snapshotCts?.Cancel();
        _snapshotCts?.Dispose();
        _snapshotCts = null;

        _isLoadingSelection = true;
        try
        {
            _committedSelection = value;
            DetachAllZones();
            Zones.Clear();
            SelectedZone = null;
            ClearUndoStack();
            if (!_preserveSnapshotOnNextSelect)
                SnapshotBytes = null;
            if (value is null)
            {
                HasUnsavedChanges = false;
                return;
            }

            EditCameraId = value.Id;
            EditName = value.Name;
            EditRtspUrl = _protector.Unprotect(value.EncryptedRtspUrl);
            EditEnabled = value.Enabled;
            EditDwellSec = value.DwellSec;
            EditReleaseWaitSec = value.ReleaseWaitSec;
            EditCooldownSec = value.CooldownSec;
            foreach (var z in value.Zones)
            {
                var ez = new EditableZoneViewModel
                {
                    Name = z.Name,
                    Role = z.Role,
                    XNorm = z.XNorm,
                    YNorm = z.YNorm,
                    WNorm = z.WNorm,
                    HNorm = z.HNorm,
                };
                AttachZone(ez);
                Zones.Add(ez);
            }
            HasUnsavedChanges = false;
        }
        finally
        {
            _isLoadingSelection = false;
        }

        if (value is not null && value.Enabled && !string.IsNullOrWhiteSpace(EditRtspUrl) && SnapshotBytes is null)
            _ = AutoGrabSnapshotAsync(value.Id, maxAttempts: 4, delayMs: 300);
    }

    [RelayCommand]
    private void NewBlankCamera()
    {
        int nextNum = Cameras.Count + 1;
        while (Cameras.Any(c => string.Equals(c.Id, $"cam_{nextNum:D2}", StringComparison.OrdinalIgnoreCase)))
            nextNum++;
        var cam = new CameraConfig
        {
            Id = $"cam_{nextNum:D2}",
            Name = $"Aisle {nextNum}",
            Enabled = true,
        };
        Cameras.Add(cam);
        SelectedCamera = cam;
        HasUnsavedChanges = true;
    }

    [RelayCommand]
    private void DiscardUnsavedChanges()
    {
        if (_committedSelection is null) return;
        ApplyCameraSelection(_committedSelection);
        StatusMessage = $"Reverted unsaved changes for {DisplayCameraLabel}.";
    }

    [RelayCommand]
    private async Task QuickAddUniviewChannelsAsync()
    {
        if (string.IsNullOrWhiteSpace(UnvIp))
        {
            StatusMessage = "Enter NVR IP address before generating channels.";
            return;
        }
        int start = Math.Max(1, UnvChannelStart);
        int end = Math.Max(start, Math.Min(start + 32, UnvChannelEnd));

        var targetIds = Enumerable.Range(start, end - start + 1).Select(ch => $"cam_{ch:D2}").ToHashSet(StringComparer.OrdinalIgnoreCase);
        int existingOverlap = Cameras.Count(c => targetIds.Contains(c.Id));
        if (existingOverlap > 0 && _confirmDialog is not null)
        {
            bool proceed = _confirmDialog(
                "Overwrite Existing Cameras",
                $"Generating Uniview channels {start}..{end} will overwrite {existingOverlap} existing camera configuration(s) and their drawn ROI zones. Continue?",
                "Overwrite & Save",
                true);
            if (!proceed) return;
        }

        for (int ch = start; ch <= end; ch++)
        {
            string cid = $"cam_{ch:D2}";
            string url = $"rtsp://{UnvUser}:{UnvPassword}@{UnvIp}:{UnvPort}/unicast/c{ch}/s1/live";
            var cam = new CameraConfig
            {
                Id = cid,
                Name = $"UNV Channel {ch}",
                EncryptedRtspUrl = _protector.Protect(url),
                Enabled = true,
            };
            await _repo.SaveCameraAsync(cam);
        }
        await LoadAsync();
        await _syncAndReloadEngineAsync();
        StatusMessage = $"Saved Uniview channels {start}–{end} and reloaded engine.";
    }

    [RelayCommand]
    private void BrowseLocalVideo()
    {
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "Video Files (*.mp4;*.mkv;*.avi)|*.mp4;*.mkv;*.avi|All Files (*.*)|*.*",
        };
        if (dlg.ShowDialog() == true)
        {
            EditRtspUrl = dlg.FileName;
        }
    }

    private async Task AutoGrabSnapshotAsync(string cameraId, int maxAttempts, int delayMs)
    {
        var cts = new CancellationTokenSource();
        _snapshotCts?.Cancel();
        _snapshotCts?.Dispose();
        _snapshotCts = cts;
        try
        {
            var bytes = await TryGrabSnapshotWithRetryAsync(cameraId, maxAttempts, delayMs, cts.Token);
            if (bytes is not null && !cts.IsCancellationRequested &&
                string.Equals(EditCameraId.Trim(), cameraId, StringComparison.OrdinalIgnoreCase))
            {
                SnapshotBytes = bytes;
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
        }
    }

    private async Task<byte[]?> TryGrabSnapshotWithRetryAsync(string cameraId, int maxAttempts, int delayMs, CancellationToken ct)
    {
        int tries = Math.Max(1, maxAttempts);
        for (int i = 0; i < tries && !ct.IsCancellationRequested; i++)
        {
            var bytes = await _client.GetSnapshotAsync(cameraId, overlays: false, ct);
            if (bytes is { Length: > 0 })
                return bytes;
            if (i + 1 < tries)
                await Task.Delay(delayMs, ct);
        }
        return null;
    }

    [RelayCommand]
    private async Task GrabSnapshotAsync()
    {
        var cid = EditCameraId.Trim();
        if (string.IsNullOrWhiteSpace(cid)) return;
        _snapshotCts?.Cancel();
        _snapshotCts?.Dispose();
        _snapshotCts = new CancellationTokenSource();

        StatusMessage = $"Grabbing snapshot for {DisplayCameraLabel}...";
        var bytes = await TryGrabSnapshotWithRetryAsync(cid, maxAttempts: 4, delayMs: 300, _snapshotCts.Token);
        if (bytes is null)
        {
            StatusMessage = $"No frame available yet for {DisplayCameraLabel}. Click 'Save Camera & Reload' first so the stream starts.";
            return;
        }
        SnapshotBytes = bytes;
        StatusMessage = $"Grabbed live frame for {DisplayCameraLabel}. Drag on frame to draw ROI zones.";
    }

    public void AddDrawnZone(double[] rectNorm)
    {
        PushZoneUndoSnapshot();
        var role = SelectedDrawRole == "aisle" ? "aisle" : "shelf";
        var baseName = string.IsNullOrWhiteSpace(NewZoneName) ? $"{role}_{Zones.Count + 1}" : NewZoneName.Trim();
        if (Zones.Any(z => z.Name.Equals(baseName, StringComparison.OrdinalIgnoreCase)))
            baseName = $"{role}_{Zones.Count + 1}";

        var zone = new EditableZoneViewModel
        {
            Name = baseName,
            Role = role,
            XNorm = rectNorm[0],
            YNorm = rectNorm[1],
            WNorm = rectNorm[2],
            HNorm = rectNorm[3],
        };
        AttachZone(zone);
        Zones.Add(zone);
        SelectedZone = zone;
        NewZoneName = $"{role}_{Zones.Count(z => string.Equals(z.Role, role, StringComparison.OrdinalIgnoreCase)) + 1}";
        HasUnsavedChanges = true;
        _logger?.Info($"ROI zone added: {baseName} ({role}) rect=[{rectNorm[0]:F2},{rectNorm[1]:F2},{rectNorm[2]:F2},{rectNorm[3]:F2}]");
    }

    public void NotifyZoneEditedFromCanvas()
    {
        HasUnsavedChanges = true;
    }

    [RelayCommand]
    private void RemoveZone(EditableZoneViewModel? zone)
    {
        if (zone is null) return;
        PushZoneUndoSnapshot();
        zone.PropertyChanged -= OnZoneItemPropertyChanged;
        Zones.Remove(zone);
        if (ReferenceEquals(SelectedZone, zone))
            SelectedZone = Zones.LastOrDefault();
        HasUnsavedChanges = true;
    }

    [RelayCommand]
    private void UndoZoneEdit()
    {
        if (_zoneUndoStack.Count == 0) return;
        var prev = _zoneUndoStack.Pop();
        OnPropertyChanged(nameof(CanUndoZoneEdit));

        DetachAllZones();
        Zones.Clear();
        foreach (var z in prev)
        {
            var ez = new EditableZoneViewModel
            {
                Name = z.Name,
                Role = z.Role,
                XNorm = z.XNorm,
                YNorm = z.YNorm,
                WNorm = z.WNorm,
                HNorm = z.HNorm,
            };
            AttachZone(ez);
            Zones.Add(ez);
        }
        SelectedZone = Zones.LastOrDefault();
        HasUnsavedChanges = true;
        StatusMessage = "Reverted last ROI zone change.";
    }

    [RelayCommand]
    private async Task ImportApexAsync()
    {
        if (string.IsNullOrWhiteSpace(EditCameraId))
            NewBlankCamera();

        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "Apex Solution (*.apex;*.xml)|*.apex;*.xml|All Files (*.*)|*.*",
        };
        if (dlg.ShowDialog() != true) return;

        var (streamUrl, solutionName, zones) = ApexProjectImporter.ParseApexFile(dlg.FileName, EditCameraId);
        ApplyParsedApex(streamUrl, solutionName, zones, System.IO.Path.GetFileName(dlg.FileName), "Click Save Camera to apply.");
    }

    public void ApplyParsedApex(
        string streamUrl,
        string solutionName,
        List<CameraZone> zones,
        string sourceLabel,
        string suffix = "Click Save Camera to persist.")
    {
        PushZoneUndoSnapshot();
        if (!string.IsNullOrWhiteSpace(streamUrl)) EditRtspUrl = streamUrl;
        if (string.IsNullOrWhiteSpace(EditName)) EditName = solutionName;
        DetachAllZones();
        Zones.Clear();
        foreach (var z in zones)
        {
            var ez = new EditableZoneViewModel
            {
                Name = z.Name,
                Role = z.Role,
                XNorm = z.XNorm,
                YNorm = z.YNorm,
                WNorm = z.WNorm,
                HNorm = z.HNorm,
            };
            AttachZone(ez);
            Zones.Add(ez);
        }
        HasUnsavedChanges = true;
        StatusMessage = $"Imported {zones.Count} zone(s) from {sourceLabel}. {suffix}";
    }

    [RelayCommand]
    private async Task SaveZonesOnlyHotReloadAsync()
    {
        var cid = EditCameraId.Trim();
        if (string.IsNullOrWhiteSpace(cid)) return;
        var entityZones = Zones.Select(z => new CameraZone
        {
            CameraId = cid,
            Name = z.Name,
            Role = z.Role,
            XNorm = z.XNorm,
            YNorm = z.YNorm,
            WNorm = z.WNorm,
            HNorm = z.HNorm,
        }).ToList();

        await _repo.ReplaceCameraZonesAsync(cid, entityZones);
        if (_writeCamerasJsonAsync is not null)
            await _writeCamerasJsonAsync();
        var dtos = entityZones.Select(RoiCoordinateConverter.ToDto).ToList();
        bool ok = await _client.UpdateCameraZonesAsync(cid, dtos, CancellationToken.None);
        _logger?.Info($"Hot-reload zones for {cid}: count={dtos.Count}, engineSynced={ok}");
        StatusMessage = ok
            ? $"Hot-reloaded {dtos.Count} ROI zone(s) live for {DisplayCameraLabel}."
            : $"Saved {dtos.Count} zone(s) for {DisplayCameraLabel} (applies on engine start).";
        HasUnsavedChanges = false;
        await LoadAsync(preserveCameraId: cid, preserveSnapshot: true);
    }

    [RelayCommand]
    private async Task SaveCameraAsync()
    {
        var cid = EditCameraId.Trim();
        if (string.IsNullOrWhiteSpace(cid)) return;
        var entity = new CameraConfig
        {
            Id = cid,
            Name = string.IsNullOrWhiteSpace(EditName) ? cid : EditName.Trim(),
            EncryptedRtspUrl = _protector.Protect(EditRtspUrl.Trim()),
            Enabled = EditEnabled,
            DwellSec = Math.Clamp(EditDwellSec, 0.5, 30.0),
            ReleaseWaitSec = Math.Clamp(EditReleaseWaitSec, 0.5, 30.0),
            CooldownSec = Math.Clamp(EditCooldownSec, 1.0, 120.0),
            Zones = Zones.Select(z => new CameraZone
            {
                CameraId = cid,
                Name = z.Name,
                Role = z.Role,
                XNorm = z.XNorm,
                YNorm = z.YNorm,
                WNorm = z.WNorm,
                HNorm = z.HNorm,
            }).ToList(),
        };
        await _repo.SaveCameraAsync(entity);
        await _syncAndReloadEngineAsync();
        _logger?.Info($"Camera saved & reloaded: id={entity.Id}, enabled={entity.Enabled}, zones={entity.Zones.Count}");
        HasUnsavedChanges = false;
        await LoadAsync(preserveCameraId: entity.Id, preserveSnapshot: true);

        if (entity.Enabled && !string.IsNullOrWhiteSpace(EditRtspUrl))
        {
            _snapshotCts?.Cancel();
            _snapshotCts?.Dispose();
            _snapshotCts = new CancellationTokenSource();
            try
            {
                var bytes = await TryGrabSnapshotWithRetryAsync(entity.Id, maxAttempts: 8, delayMs: 350, _snapshotCts.Token);
                if (bytes is not null && string.Equals(EditCameraId.Trim(), entity.Id, StringComparison.OrdinalIgnoreCase))
                    SnapshotBytes = bytes;
            }
            catch (OperationCanceledException)
            {
            }
        }

        StatusMessage = SnapshotBytes is not null
            ? $"Saved {entity.Name}, reloaded AI engine, and refreshed preview."
            : $"Saved {entity.Name} and reloaded AI engine.";
    }

    [RelayCommand]
    private async Task DeleteCameraAsync()
    {
        if (SelectedCamera is null) return;
        var cid = SelectedCamera.Id;
        var label = string.IsNullOrWhiteSpace(SelectedCamera.Name) ? cid : SelectedCamera.Name;
        if (_confirmDialog is not null)
        {
            bool confirmed = _confirmDialog(
                "Delete Camera",
                $"Delete '{label}' and all {Zones.Count} drawn ROI zone(s)? This immediately reloads the AI engine.",
                "Delete Camera",
                true);
            if (!confirmed) return;
        }

        await _repo.DeleteCameraAsync(cid);
        HasUnsavedChanges = false;
        await LoadAsync();
        await _syncAndReloadEngineAsync();
        _logger?.Info($"Camera deleted: id={cid}");
        StatusMessage = $"Deleted {label}.";
    }
}
