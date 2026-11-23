using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;

namespace StoreGuard.Desktop.ViewModels;

public sealed partial class LiveGridViewModel : ObservableObject
{
    private readonly IEngineClient _client;
    private readonly IEventRepository _repo;
    private readonly Dictionary<string, bool> _selectionMemory = new(StringComparer.OrdinalIgnoreCase);
    private volatile CameraSnapshotTileViewModel[] _visibleSnapshot = Array.Empty<CameraSnapshotTileViewModel>();
    private CancellationTokenSource? _pollCts;
    private Task? _pollTask;
    private string _layoutMode = "auto";

    public ObservableCollection<CameraSnapshotTileViewModel> AllTiles { get; } = new();
    public ObservableCollection<CameraSnapshotTileViewModel> VisibleTiles { get; } = new();
    public ObservableCollection<CameraSnapshotTileViewModel> Tiles => VisibleTiles;

    [ObservableProperty] private bool _isLayoutAuto = true;
    [ObservableProperty] private bool _isCols2;
    [ObservableProperty] private bool _isCols3;
    [ObservableProperty] private bool _isCols4;
    [ObservableProperty] private int _effectiveColumns = 1;
    [ObservableProperty] private int _effectiveRows = 1;
    [ObservableProperty] private double _viewportWidth = 960.0;
    [ObservableProperty] private double _viewportHeight = 540.0;
    [ObservableProperty] private bool _hasCameras;
    [ObservableProperty] private bool _hasAnyCameras;
    [ObservableProperty] private bool _hasVisibleCameras;
    [ObservableProperty] private bool _allCamerasHidden;
    [ObservableProperty] private bool _hasMultipleCameras;
    [ObservableProperty] private string _selectionSummary = "0 enabled cameras";
    [ObservableProperty] private bool _showRoiOverlays = true;
    [ObservableProperty] private bool _showShelfOverlay = true;
    [ObservableProperty] private bool _showAisleOverlay = true;
    [ObservableProperty] private bool _showPoseOverlay = true;
    [ObservableProperty] private bool _showBotSortOverlay = true;
    [ObservableProperty] private bool _showBagsOverlay = true;
    [ObservableProperty] private bool _showMotionOverlay = false;
    [ObservableProperty] private CameraSnapshotTileViewModel? _focusedTile;

    public string GetActiveOverlayLayers()
    {
        if (!ShowRoiOverlays)
            return string.Empty;
        var layers = new List<string>(6);
        if (ShowShelfOverlay) layers.Add("shelf");
        if (ShowAisleOverlay) layers.Add("aisle");
        if (ShowPoseOverlay) layers.Add("pose");
        if (ShowBotSortOverlay) layers.Add("botsort");
        if (ShowBagsOverlay) layers.Add("bags");
        if (ShowMotionOverlay) layers.Add("motion");
        return string.Join(",", layers);
    }
    [ObservableProperty] private int _currentPage = 1;
    [ObservableProperty] private int _totalPages = 1;
    [ObservableProperty] private bool _hasMultiplePages;
    [ObservableProperty] private bool _canGoPreviousPage;
    [ObservableProperty] private bool _canGoNextPage;
    [ObservableProperty] private string _pageSummary = "Page 1 of 1";

    public bool HasFocusedTile => FocusedTile is not null;
    public int PageSize => _layoutMode switch
    {
        "2" => 4,
        "3" => 9,
        _ => 16,
    };

    partial void OnFocusedTileChanged(CameraSnapshotTileViewModel? value) => OnPropertyChanged(nameof(HasFocusedTile));

    public LiveGridViewModel(IEngineClient client, IEventRepository repo)
    {
        _client = client;
        _repo = repo;
    }

    [RelayCommand]
    private void SetGridLayout(string? mode)
    {
        var normalized = string.IsNullOrWhiteSpace(mode) ? "auto" : mode.Trim().ToLowerInvariant();
        if (normalized is not ("auto" or "2" or "3" or "4"))
            normalized = "auto";
        _layoutMode = normalized;
        IsLayoutAuto = normalized == "auto";
        IsCols2 = normalized == "2";
        IsCols3 = normalized == "3";
        IsCols4 = normalized == "4";
        CurrentPage = 1;
        RebuildVisibleTiles();
    }

    [RelayCommand]
    private void PreviousPage()
    {
        if (CurrentPage > 1)
        {
            CurrentPage--;
            RebuildVisibleTiles();
        }
    }

    [RelayCommand]
    private void NextPage()
    {
        if (CurrentPage < TotalPages)
        {
            CurrentPage++;
            RebuildVisibleTiles();
        }
    }

    [RelayCommand]
    private void SelectAllCameras()
    {
        foreach (var tile in AllTiles)
            tile.IsSelected = true;
    }

    [RelayCommand]
    private void ClearSelectedCameras()
    {
        foreach (var tile in AllTiles)
            tile.IsSelected = false;
    }

    [RelayCommand]
    private async Task RerunCameraAsync(CameraSnapshotTileViewModel? tile)
    {
        if (tile is null) return;
        tile.IsRerunning = true;
        try
        {
            var ok = await _client.RerunCameraDetectionAsync(tile.CameraId, CancellationToken.None);
            if (ok)
                tile.ClipDetectionDone = false;
        }
        finally
        {
            tile.IsRerunning = false;
        }
    }

    [RelayCommand]
    private void FocusTile(CameraSnapshotTileViewModel? tile)
    {
        if (tile is not null)
            FocusedTile = tile;
    }

    [RelayCommand]
    private void ExitFocus() => FocusedTile = null;

    public void UpdateViewportSize(double width, double height)
    {
        if (width <= 0 || height <= 0 || double.IsNaN(width) || double.IsNaN(height))
            return;
        if (Math.Abs(width - ViewportWidth) < 2.0 && Math.Abs(height - ViewportHeight) < 2.0)
            return;
        ViewportWidth = width;
        ViewportHeight = height;
        RecalculateLayout();
    }

    public void ApplyHealthEvent(CameraHealthEvent health)
    {
        var tile = AllTiles.FirstOrDefault(t => string.Equals(t.CameraId, health.CameraId, StringComparison.OrdinalIgnoreCase));
        if (tile is null) return;
        bool wasLocal = tile.IsLocalFile;
        tile.State = health.State;
        tile.Fps = health.Fps;
        tile.IsLocalFile = health.IsLocalFile;
        tile.ClipDetectionDone = health.ClipDetectionDone;
        if (wasLocal != tile.IsLocalFile)
            RecalculateLayout();
    }

    public async Task StartAsync()
    {
        Stop();
        var cams = await _repo.GetCamerasAsync();
        foreach (var oldTile in AllTiles)
            oldTile.PropertyChanged -= OnTilePropertyChanged;
        AllTiles.Clear();

        foreach (var c in cams.Where(x => x.Enabled))
        {
            bool selected = !_selectionMemory.TryGetValue(c.Id, out var prev) || prev;
            var tile = new CameraSnapshotTileViewModel
            {
                CameraId = c.Id,
                Name = c.Name,
                IsSelected = selected,
            };
            tile.PropertyChanged += OnTilePropertyChanged;
            AllTiles.Add(tile);
        }

        RebuildVisibleTiles();

        _pollCts = new CancellationTokenSource();
        _pollTask = Task.Run(() => PollLoopAsync(_pollCts.Token), _pollCts.Token);
    }

    public void Stop()
    {
        _pollCts?.Cancel();
        _pollCts?.Dispose();
        _pollCts = null;
        _pollTask = null;
        _visibleSnapshot = Array.Empty<CameraSnapshotTileViewModel>();
        FocusedTile = null;
    }

    private void OnTilePropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (sender is CameraSnapshotTileViewModel tile && e.PropertyName == nameof(CameraSnapshotTileViewModel.IsSelected))
        {
            _selectionMemory[tile.CameraId] = tile.IsSelected;
            RebuildVisibleTiles();
        }
    }

    public void RebuildVisibleTiles()
    {
        var selected = AllTiles.Where(t => t.IsSelected).ToList();
        int pageSize = Math.Max(1, PageSize);
        int pages = Math.Max(1, (int)Math.Ceiling(selected.Count / (double)pageSize));
        int page = Math.Clamp(CurrentPage, 1, pages);
        if (page != CurrentPage)
            CurrentPage = page;

        TotalPages = pages;
        HasMultiplePages = pages > 1;
        CanGoPreviousPage = page > 1;
        CanGoNextPage = page < pages;
        PageSummary = $"Page {page} of {pages}";

        VisibleTiles.Clear();
        foreach (var t in selected.Skip((page - 1) * pageSize).Take(pageSize))
            VisibleTiles.Add(t);
        _visibleSnapshot = VisibleTiles.ToArray();

        if (FocusedTile is not null && !FocusedTile.IsSelected)
            FocusedTile = null;

        HasAnyCameras = AllTiles.Count > 0;
        HasCameras = HasAnyCameras;
        HasVisibleCameras = VisibleTiles.Count > 0;
        AllCamerasHidden = HasAnyCameras && selected.Count == 0;
        HasMultipleCameras = AllTiles.Count > 1;
        SelectionSummary = AllTiles.Count switch
        {
            0 => "0 enabled cameras",
            1 => selected.Count == 1 ? "1 camera live" : "0 of 1 camera selected",
            _ when pages > 1 => $"{selected.Count} of {AllTiles.Count} cameras · Showing {VisibleTiles.Count} (Page {page}/{pages})",
            _ => $"{selected.Count} of {AllTiles.Count} cameras selected",
        };

        RecalculateLayout();
    }

    public void RecalculateLayout()
    {
        int count = VisibleTiles.Count;
        if (count <= 0)
        {
            EffectiveColumns = 1;
            EffectiveRows = 1;
            return;
        }

        int cols = _layoutMode switch
        {
            "2" => 2,
            "3" => 3,
            "4" => 4,
            _ => count == 1 ? 1 : (count <= 4 ? 2 : (count <= 9 ? 3 : 4)),
        };
        EffectiveColumns = cols;
        int rows = Math.Max(1, (int)Math.Ceiling(count / (double)cols));
        EffectiveRows = rows;

        double availW = Math.Max(260.0, ViewportWidth - 14.0);
        double availH = Math.Max(200.0, ViewportHeight - 8.0);
        double cellGap = 10.0;

        double cellMaxW = Math.Max(180.0, (availW - cellGap * (cols - 1)) / cols);
        int rowsForFit = Math.Min(rows, 3);
        double cellMaxH = Math.Max(165.0, (availH - cellGap * (rowsForFit - 1)) / rowsForFit);

        foreach (var tile in VisibleTiles)
        {
            double chromeW = 28.0;
            double chromeH = tile.IsLocalFile ? 92.0 : 60.0;
            double maxMediaW = Math.Max(110.0, cellMaxW - chromeW);
            double maxMediaH = Math.Max(90.0, cellMaxH - chromeH);
            double ar = Math.Clamp(tile.FeedAspectRatio, 0.45, 2.60);

            double fitW = Math.Min(maxMediaW, maxMediaH * ar);
            double fitH = fitW / ar;

            tile.TileMediaWidth = Math.Floor(fitW);
            tile.TileMediaHeight = Math.Floor(fitH);
            tile.TileMaxWidth = Math.Floor(Math.Max(fitW, Math.Min(maxMediaW, 240.0)) + 18.0);
        }
    }

    public static BitmapImage? DecodeFrozenBitmap(byte[]? bytes)
    {
        if (bytes is null || bytes.Length == 0)
            return null;
        try
        {
            using var ms = new MemoryStream(bytes, writable: false);
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.CreateOptions = BitmapCreateOptions.IgnoreColorProfile;
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

    public static (int MaxWidth, int Quality, int TargetPeriodMs) ComputeStreamProfile(bool isFocused, int visibleCount)
    {
        if (isFocused)
            return (1280, 80, 75);
        if (visibleCount <= 1)
            return (960, 78, 80);
        if (visibleCount <= 4)
            return (640, 75, 90);
        if (visibleCount <= 9)
            return (480, 72, 110);
        return (420, 70, 130);
    }

    private static void DispatchToUi(Action apply)
    {
        var disp = System.Windows.Application.Current?.Dispatcher;
        if (disp is not null && !disp.HasShutdownStarted && !disp.HasShutdownFinished && disp.Thread.IsAlive)
        {
            if (disp.CheckAccess())
                apply();
            else
                _ = disp.InvokeAsync(apply, DispatcherPriority.Render);
        }
        else
        {
            apply();
        }
    }

    private async Task<EngineHealthResponse?> FetchHealthBoundedAsync(CancellationToken ct)
    {
        try
        {
            using var reqCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            reqCts.CancelAfter(1200);
            return await _client.GetHealthAsync(reqCts.Token);
        }
        catch
        {
            return null;
        }
    }

    private async Task FetchDecodeAndApplyTileAsync(
        CameraSnapshotTileViewModel tile,
        bool overlays,
        string? overlayLayers,
        int maxWidth,
        int quality,
        SemaphoreSlim fetchGate,
        CancellationToken ct)
    {
        bool acquired = false;
        try
        {
            await fetchGate.WaitAsync(ct);
            acquired = true;
            using var reqCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            reqCts.CancelAfter(1200);
            var jpg = await _client.GetSnapshotAsync(tile.CameraId, overlays: overlays, reqCts.Token, maxWidth, quality, overlayLayers);
            if (jpg is null || ct.IsCancellationRequested)
                return;

            var bmp = DecodeFrozenBitmap(jpg);
            if (bmp is null || ct.IsCancellationRequested)
                return;

            int imgW = bmp.PixelWidth;
            int imgH = bmp.PixelHeight;
            DispatchToUi(() =>
            {
                if (ct.IsCancellationRequested)
                    return;
                tile.SnapshotImage = bmp;
                if (imgW > 0 && imgH > 0 && tile.UpdateFeedDimensions(imgW, imgH))
                    RecalculateLayout();
            });
        }
        catch
        {
            // Transient snapshot timeout/cancellation — next tick retries automatically
        }
        finally
        {
            if (acquired)
                fetchGate.Release();
        }
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        var healthSw = Stopwatch.StartNew();
        bool healthPrimed = false;
        using var fetchGate = new SemaphoreSlim(6, 6);

        while (!ct.IsCancellationRequested)
        {
            var tickSw = Stopwatch.StartNew();
            var activeLayers = GetActiveOverlayLayers();
            bool overlays = ShowRoiOverlays && !string.IsNullOrEmpty(activeLayers);
            var focused = FocusedTile;
            var targets = focused is not null ? new[] { focused } : _visibleSnapshot;
            var (maxWidth, quality, targetPeriodMs) = ComputeStreamProfile(focused is not null, targets.Length);

            Task<EngineHealthResponse?>? healthTask = (!healthPrimed || healthSw.ElapsedMilliseconds >= 1500)
                ? FetchHealthBoundedAsync(ct)
                : null;

            if (targets.Length > 0)
            {
                await Task.WhenAll(targets.Select(tile =>
                    FetchDecodeAndApplyTileAsync(tile, overlays, activeLayers, maxWidth, quality, fetchGate, ct)));
            }

            if (healthTask is not null)
            {
                var health = await healthTask;
                if (health is not null && !ct.IsCancellationRequested)
                {
                    var healthMap = health.Cameras.ToDictionary(c => c.CameraId, StringComparer.OrdinalIgnoreCase);
                    healthPrimed = true;
                    healthSw.Restart();
                    DispatchToUi(() =>
                    {
                        if (ct.IsCancellationRequested)
                            return;
                        bool layoutDirty = false;
                        foreach (var tile in targets)
                        {
                            if (!healthMap.TryGetValue(tile.CameraId, out var h))
                                continue;
                            bool localChanged = tile.IsLocalFile != h.IsLocalFile;
                            tile.State = h.State;
                            tile.Fps = h.Fps;
                            tile.IsLocalFile = h.IsLocalFile;
                            tile.ClipDetectionDone = h.ClipDetectionDone;
                            if (localChanged)
                                layoutDirty = true;
                        }
                        if (layoutDirty)
                            RecalculateLayout();
                    });
                }
            }

            int remainMs = Math.Max(15, targetPeriodMs - (int)tickSw.ElapsedMilliseconds);
            try { await Task.Delay(remainMs, ct); } catch { break; }
        }
    }

    public static bool TryReadJpegDimensions(byte[] data, out int width, out int height)
    {
        width = 0;
        height = 0;
        if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8)
            return false;

        int i = 2;
        while (i + 3 < data.Length)
        {
            while (i < data.Length && data[i] != 0xFF) i++;
            while (i < data.Length && data[i] == 0xFF) i++;
            if (i >= data.Length) break;

            byte marker = data[i++];
            if (marker is 0xD8 or 0xD9 or 0x01 || (marker >= 0xD0 && marker <= 0xD7))
                continue;
            if (i + 1 >= data.Length) break;

            int segLen = (data[i] << 8) | data[i + 1];
            if (segLen < 2 || i + segLen > data.Length)
                break;

            if (marker is 0xC0 or 0xC1 or 0xC2 or 0xC3 or 0xC5 or 0xC6 or 0xC7 or 0xC9 or 0xCA or 0xCB or 0xCD or 0xCE or 0xCF)
            {
                if (i + 6 < data.Length)
                {
                    height = (data[i + 3] << 8) | data[i + 4];
                    width = (data[i + 5] << 8) | data[i + 6];
                    return width > 0 && height > 0;
                }
                break;
            }
            i += segLen;
        }
        return false;
    }
}
