using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using LibVLCSharp.Shared;
using StoreGuard.Desktop.ViewModels;

namespace StoreGuard.Desktop.Views;

public partial class StaffDashboardView : UserControl
{
    private LibVLC? _libVlc;
    private MediaPlayer? _player;
    private AlertFeedViewModel? _boundVm;

    public StaffDashboardView()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
        Unloaded += OnUnloaded;
    }

    private void OnDataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
    {
        if (_boundVm is not null)
            _boundVm.ClipPlaybackRequested -= PlayClipFile;

        _boundVm = e.NewValue as AlertFeedViewModel;
        if (_boundVm is not null)
            _boundVm.ClipPlaybackRequested += PlayClipFile;
    }

    private void UserControl_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (_boundVm is null || Keyboard.Modifiers != ModifierKeys.None) return;
        if (Keyboard.FocusedElement is TextBoxBase or PasswordBox) return;

        switch (e.Key)
        {
            case Key.F:
                if (_boundVm.MarkFalseAlarmCommand.CanExecute(null))
                {
                    _boundVm.MarkFalseAlarmCommand.Execute(null);
                    e.Handled = true;
                }
                break;
            case Key.C:
                if (_boundVm.ConfirmTheftCommand.CanExecute(null))
                {
                    _boundVm.ConfirmTheftCommand.Execute(null);
                    e.Handled = true;
                }
                break;
            case Key.Space when Keyboard.FocusedElement is not ButtonBase:
                if (_boundVm.ToggleMediaModeCommand.CanExecute(null))
                {
                    _boundVm.ToggleMediaModeCommand.Execute(null);
                    e.Handled = true;
                }
                break;
            case Key.Up when Keyboard.FocusedElement is not ListBoxItem:
                _boundVm.SelectPreviousAlertCommand.Execute(null);
                e.Handled = true;
                break;
            case Key.Down when Keyboard.FocusedElement is not ListBoxItem:
                _boundVm.SelectNextAlertCommand.Execute(null);
                e.Handled = true;
                break;
        }
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
                ClipVideoView.MediaPlayer = _player;
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
