using System.Collections.ObjectModel;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Infrastructure.Persistence;

namespace StoreGuard.Desktop.ViewModels;

public sealed partial class HistoryViewModel : ObservableObject
{
    private readonly IEventRepository _repo;
    private List<EventRecord> _lastRecords = new();
    private bool _suppressAutoRefresh;
    private CancellationTokenSource? _autoRefreshCts;

    public ObservableCollection<AlertItemViewModel> Items { get; } = new();
    public ObservableCollection<FilterOption> CameraFilters { get; } = new() { new("ALL", "All Cameras") };
    public IReadOnlyList<FilterOption> VerdictFilters { get; } =
    [
        new("All", "All Verdicts"),
        new("Unreviewed", "Unreviewed"),
        new("ConfirmedTheft", "Confirmed Theft"),
        new("FalseAlarm", "False Alarm"),
    ];

    [ObservableProperty] private string _selectedCameraFilter = "ALL";
    [ObservableProperty] private string _selectedVerdictFilter = "All";
    [ObservableProperty] private DateTime? _fromDate = DateTime.Today.AddDays(-7);
    [ObservableProperty] private DateTime? _toDate = DateTime.Today;
    [ObservableProperty] private AlertItemViewModel? _selectedItem;
    [ObservableProperty] private string _statusText = string.Empty;
    [ObservableProperty] private bool _showClipMode;
    [ObservableProperty] private string _reviewNoteInput = string.Empty;

    public bool IsGridMode => !ShowClipMode;
    public bool CanPlaySelectedClip => SelectedItem?.HasValidClip == true;

    public event Action<string>? ClipPlaybackRequested;
    public event Action<string, string, string, string, string>? ReviewUpdated;

    public HistoryViewModel(IEventRepository repo)
    {
        _repo = repo;
    }

    partial void OnShowClipModeChanged(bool value) => OnPropertyChanged(nameof(IsGridMode));

    partial void OnSelectedItemChanged(AlertItemViewModel? value)
    {
        ReviewNoteInput = value?.ReviewNote ?? string.Empty;
        ShowClipMode = false;
        OnPropertyChanged(nameof(CanPlaySelectedClip));
    }

    partial void OnSelectedCameraFilterChanged(string value) => ScheduleAutoRefresh();
    partial void OnSelectedVerdictFilterChanged(string value) => ScheduleAutoRefresh();
    partial void OnFromDateChanged(DateTime? value) => ScheduleAutoRefresh();
    partial void OnToDateChanged(DateTime? value) => ScheduleAutoRefresh();

    private void ScheduleAutoRefresh()
    {
        if (_suppressAutoRefresh) return;
        _autoRefreshCts?.Cancel();
        _autoRefreshCts?.Dispose();
        var cts = new CancellationTokenSource();
        _autoRefreshCts = cts;
        _ = DebouncedRefreshAsync(cts.Token);
    }

    private async Task DebouncedRefreshAsync(CancellationToken ct)
    {
        try
        {
            await Task.Delay(120, ct);
            if (!ct.IsCancellationRequested)
                await RefreshAsync();
        }
        catch (OperationCanceledException)
        {
        }
    }

    [RelayCommand]
    public async Task RefreshAsync()
    {
        _suppressAutoRefresh = true;
        try
        {
            var cams = await _repo.GetCamerasAsync();
            CameraFilters.Clear();
            CameraFilters.Add(new FilterOption("ALL", "All Cameras"));
            foreach (var c in cams)
            {
                var label = string.IsNullOrWhiteSpace(c.Name) ? c.Id : $"{c.Name} ({c.Id})";
                CameraFilters.Add(new FilterOption(c.Id, label));
            }
            if (!CameraFilters.Any(x => string.Equals(x.Value, SelectedCameraFilter, StringComparison.Ordinal)))
                SelectedCameraFilter = "ALL";

            DateTime? fromUtc = FromDate?.Date.ToUniversalTime();
            DateTime? toUtc = ToDate.HasValue
                ? ToDate.Value.Date.AddDays(1).AddTicks(-1).ToUniversalTime()
                : null;
            _lastRecords = await _repo.QueryHistoryAsync(fromUtc, toUtc, SelectedCameraFilter, SelectedVerdictFilter);

            var prevSelectedId = SelectedItem?.EventId;
            Items.Clear();
            foreach (var ev in _lastRecords)
            {
                Items.Add(AlertItemViewModel.FromRecord(ev));
            }
            SelectedItem = Items.FirstOrDefault(i => i.EventId == prevSelectedId) ?? Items.FirstOrDefault();
            StatusText = $"{Items.Count} record(s) found";
        }
        finally
        {
            _suppressAutoRefresh = false;
        }
    }

    [RelayCommand]
    private void ToggleGridView() => ShowClipMode = false;

    [RelayCommand]
    private void PlayClip()
    {
        if (SelectedItem is null || !SelectedItem.HasValidClip)
        {
            ShowClipMode = false;
            return;
        }
        ShowClipMode = true;
        ClipPlaybackRequested?.Invoke(SelectedItem.ClipPath);
    }

    [RelayCommand]
    private Task ConfirmTheftAsync() => SubmitVerdictAsync("ConfirmedTheft");

    [RelayCommand]
    private Task MarkFalseAlarmAsync() => SubmitVerdictAsync("FalseAlarm");

    private async Task SubmitVerdictAsync(string verdict)
    {
        if (SelectedItem is null) return;
        await _repo.SubmitReviewAsync(SelectedItem.EventId, verdict, ReviewNoteInput, "CounterStaff", null);
        SelectedItem.Verdict = verdict;
        SelectedItem.ReviewNote = ReviewNoteInput;
        SelectedItem.ReviewedBy = "CounterStaff";
        SelectedItem.ReviewedAtDisplay = DateTime.Now.ToString("yyyy-MM-dd HH:mm");
        StatusText = $"Updated {SelectedItem.CameraName} verdict to {SelectedItem.VerdictDisplay}";
        ReviewUpdated?.Invoke(SelectedItem.EventId, verdict, SelectedItem.ReviewNote, SelectedItem.ReviewedBy, SelectedItem.ReviewedAtDisplay);
    }

    [RelayCommand]
    private void ExportCsv()
    {
        var dlg = new Microsoft.Win32.SaveFileDialog
        {
            Filter = "CSV File (*.csv)|*.csv",
            FileName = $"StoreGuard_Audit_{DateTime.Now:yyyyMMdd_HHmm}.csv",
        };
        if (dlg.ShowDialog() != true) return;
        var csv = EventRepository.BuildCsvReport(_lastRecords);
        File.WriteAllText(dlg.FileName, csv);
        StatusText = $"Exported {_lastRecords.Count} records to {Path.GetFileName(dlg.FileName)}";
    }
}
