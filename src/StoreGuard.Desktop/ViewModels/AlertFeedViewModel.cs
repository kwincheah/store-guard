using System.Collections.ObjectModel;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;

namespace StoreGuard.Desktop.ViewModels;

public sealed partial class AlertFeedViewModel : ObservableObject
{
    private readonly IEventRepository _repo;
    private CancellationTokenSource? _undoCts;
    private string _undoEventId = string.Empty;
    private string _undoPrevVerdict = "Unreviewed";
    private string _undoPrevNote = string.Empty;
    private string _undoPrevReviewedBy = string.Empty;
    private string _undoPrevReviewedAt = string.Empty;

    public ObservableCollection<AlertItemViewModel> Alerts { get; } = new();
    public ObservableCollection<CameraHealthPillViewModel> CameraHealths { get; } = new();

    [ObservableProperty] private AlertItemViewModel? _selectedAlert;
    [ObservableProperty] private string _reviewNoteInput = string.Empty;
    [ObservableProperty] private bool _showClipMode;
    [ObservableProperty] private bool _hasCameras;
    [ObservableProperty] private int _unreviewedCount;
    [ObservableProperty] private bool _showUndoToast;
    [ObservableProperty] private string _undoToastMessage = string.Empty;

    public event Action<string>? ClipPlaybackRequested;

    public AlertFeedViewModel(IEventRepository repo)
    {
        _repo = repo;
    }

    public bool IsGridMode => !ShowClipMode;
    public bool HasSelectedAlert => SelectedAlert is not null;
    public bool CanPlaySelectedClip => SelectedAlert?.HasValidClip == true;

    partial void OnShowClipModeChanged(bool value) => OnPropertyChanged(nameof(IsGridMode));

    partial void OnSelectedAlertChanged(AlertItemViewModel? value)
    {
        ReviewNoteInput = value?.ReviewNote ?? string.Empty;
        ShowClipMode = false;
        OnPropertyChanged(nameof(HasSelectedAlert));
        OnPropertyChanged(nameof(CanPlaySelectedClip));
    }

    public async Task LoadInitialAsync()
    {
        var cameras = await _repo.GetCamerasAsync();
        CameraHealths.Clear();
        foreach (var cam in cameras)
        {
            CameraHealths.Add(new CameraHealthPillViewModel
            {
                CameraId = cam.Id,
                Name = cam.Name,
                State = cam.Enabled ? "OFFLINE" : "DISABLED",
            });
        }

        var recent = await _repo.GetRecentEventsAsync(80);
        Alerts.Clear();
        foreach (var ev in recent)
        {
            Alerts.Add(AlertItemViewModel.FromRecord(ev));
        }
        HasCameras = CameraHealths.Count > 0;
        UnreviewedCount = Alerts.Count(a => a.Verdict == "Unreviewed");
        SelectedAlert = Alerts.FirstOrDefault();
    }

    public void AddIncomingAlert(TheftAlertEvent alert)
    {
        var existing = Alerts.FirstOrDefault(a => a.EventId == alert.EventId);
        if (existing is not null) return;

        var vm = new AlertItemViewModel
        {
            EventId = alert.EventId,
            CameraId = alert.CameraId,
            CameraName = alert.CameraName,
            TrackId = alert.TrackId,
            BehaviorCategory = alert.BehaviorCategory,
            ConfidencePct = alert.ConfidencePct,
            BehaviorsSummary = string.Join(", ", alert.Behaviors),
            Justification = alert.Justification,
            GridPath = alert.GridPath,
            ClipPath = alert.ClipPath,
            CreatedAtUtc = alert.TimestampUtc,
        };
        Alerts.Insert(0, vm);
        UnreviewedCount = Alerts.Count(a => a.Verdict == "Unreviewed");
        SelectedAlert = vm;
    }

    public void UpdateCameraHealth(CameraHealthEvent health)
    {
        var pill = CameraHealths.FirstOrDefault(c => c.CameraId == health.CameraId);
        if (pill is null)
        {
            pill = new CameraHealthPillViewModel { CameraId = health.CameraId, Name = health.CameraId };
            CameraHealths.Add(pill);
            HasCameras = true;
        }
        pill.State = health.State;
        pill.Fps = health.Fps;
        pill.MotionActive = health.MotionActive;
        pill.Error = health.Error;
    }

    public void SyncReviewFromHistory(string eventId, string verdict, string note, string reviewedBy, string reviewedAtDisplay)
    {
        var match = Alerts.FirstOrDefault(a => a.EventId == eventId);
        if (match is null) return;
        match.Verdict = verdict;
        match.ReviewNote = note;
        match.ReviewedBy = reviewedBy;
        match.ReviewedAtDisplay = reviewedAtDisplay;
        if (ReferenceEquals(SelectedAlert, match))
            ReviewNoteInput = note;
        UnreviewedCount = Alerts.Count(a => a.Verdict == "Unreviewed");
    }

    [RelayCommand]
    private Task ConfirmTheftAsync() => SubmitVerdictAsync("ConfirmedTheft");

    [RelayCommand]
    private Task MarkFalseAlarmAsync() => SubmitVerdictAsync("FalseAlarm");

    [RelayCommand]
    private void ToggleGridView() => ShowClipMode = false;

    [RelayCommand]
    private void PlayClip()
    {
        if (SelectedAlert is null || !SelectedAlert.HasValidClip)
        {
            ShowClipMode = false;
            return;
        }
        ShowClipMode = true;
        ClipPlaybackRequested?.Invoke(SelectedAlert.ClipPath);
    }

    [RelayCommand]
    private void ToggleMediaMode()
    {
        if (ShowClipMode)
            ToggleGridView();
        else if (CanPlaySelectedClip)
            PlayClip();
    }

    [RelayCommand]
    private void SelectPreviousAlert()
    {
        if (Alerts.Count == 0) return;
        int idx = SelectedAlert is null ? 0 : Alerts.IndexOf(SelectedAlert);
        if (idx > 0)
            SelectedAlert = Alerts[idx - 1];
    }

    [RelayCommand]
    private void SelectNextAlert()
    {
        if (Alerts.Count == 0) return;
        int idx = SelectedAlert is null ? -1 : Alerts.IndexOf(SelectedAlert);
        if (idx + 1 < Alerts.Count)
            SelectedAlert = Alerts[idx + 1];
    }

    [RelayCommand]
    private async Task UndoLastVerdictAsync()
    {
        if (string.IsNullOrWhiteSpace(_undoEventId)) return;
        _undoCts?.Cancel();
        ShowUndoToast = false;

        var target = Alerts.FirstOrDefault(a => a.EventId == _undoEventId);
        if (_undoPrevVerdict == "Unreviewed")
        {
            await _repo.DeleteReviewAsync(_undoEventId);
            if (target is not null)
            {
                target.Verdict = "Unreviewed";
                target.ReviewNote = _undoPrevNote;
                target.ReviewedBy = string.Empty;
                target.ReviewedAtDisplay = string.Empty;
            }
        }
        else
        {
            var reviewer = string.IsNullOrWhiteSpace(_undoPrevReviewedBy) ? "CounterStaff" : _undoPrevReviewedBy;
            await _repo.SubmitReviewAsync(_undoEventId, _undoPrevVerdict, _undoPrevNote, reviewer, null);
            if (target is not null)
            {
                target.Verdict = _undoPrevVerdict;
                target.ReviewNote = _undoPrevNote;
                target.ReviewedBy = reviewer;
                target.ReviewedAtDisplay = _undoPrevReviewedAt;
            }
        }

        if (target is not null && ReferenceEquals(SelectedAlert, target))
            ReviewNoteInput = target.ReviewNote;

        UnreviewedCount = Alerts.Count(a => a.Verdict == "Unreviewed");
        _undoEventId = string.Empty;
    }

    private async Task SubmitVerdictAsync(string verdict)
    {
        if (SelectedAlert is null) return;
        var target = SelectedAlert;

        _undoEventId = target.EventId;
        _undoPrevVerdict = target.Verdict;
        _undoPrevNote = target.ReviewNote;
        _undoPrevReviewedBy = target.ReviewedBy;
        _undoPrevReviewedAt = target.ReviewedAtDisplay;

        await _repo.SubmitReviewAsync(target.EventId, verdict, ReviewNoteInput, "CounterStaff", null);
        target.Verdict = verdict;
        target.ReviewNote = ReviewNoteInput;
        target.ReviewedBy = "CounterStaff";
        target.ReviewedAtDisplay = DateTime.Now.ToString("yyyy-MM-dd HH:mm");
        UnreviewedCount = Alerts.Count(a => a.Verdict == "Unreviewed");

        UndoToastMessage = verdict == "ConfirmedTheft"
            ? $"Marked '{target.CameraName}' as Confirmed Theft"
            : $"Marked '{target.CameraName}' as False Alarm";
        ShowUndoToast = true;

        _undoCts?.Cancel();
        _undoCts?.Dispose();
        var cts = new CancellationTokenSource();
        _undoCts = cts;
        _ = HideUndoToastLaterAsync(cts.Token);
    }

    private async Task HideUndoToastLaterAsync(CancellationToken ct)
    {
        try
        {
            await Task.Delay(5000, ct);
            if (!ct.IsCancellationRequested)
                ShowUndoToast = false;
        }
        catch (OperationCanceledException)
        {
        }
    }
}
