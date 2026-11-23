using System.Windows;
using System.Windows.Controls;
using LibVLCSharp.Shared;
using StoreGuard.Desktop.ViewModels;

namespace StoreGuard.Desktop.Views;

public partial class HistoryView : UserControl
{
    private LibVLC? _libVlc;
    private MediaPlayer? _player;
    private HistoryViewModel? _boundVm;

    public HistoryView()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
        Unloaded += OnUnloaded;
    }

    private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
    {
        if (_boundVm is not null)
            _boundVm.ClipPlaybackRequested -= PlayClipFile;

        _boundVm = e.NewValue as HistoryViewModel;
        if (_boundVm is not null)
            _boundVm.ClipPlaybackRequested += PlayClipFile;
    }

    private void PlayClipFile(string clipPath)
    {
        try
        {
            if (_libVlc is null)
            {
                LibVLCSharp.Shared.Core.Initialize();
                _libVlc = new LibVLC();
                _player = new MediaPlayer(_libVlc);
                HistoryClipVideoView.MediaPlayer = _player;
            }
            if (_player is null || _libVlc is null) return;
            using var media = new Media(_libVlc, clipPath, FromType.FromPath);
            _player.Play(media);
        }
        catch
        {
            // Keep UI stable if VideoLAN native codecs are unavailable
        }
    }

    private void OnUnloaded(object sender, RoutedEventArgs e)
    {
        try
        {
            _player?.Stop();
            _player?.Dispose();
            _libVlc?.Dispose();
        }
        catch { }
        _player = null;
        _libVlc = null;
    }
}
