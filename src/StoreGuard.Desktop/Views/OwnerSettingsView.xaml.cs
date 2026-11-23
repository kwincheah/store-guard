using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using StoreGuard.Desktop.ViewModels;

namespace StoreGuard.Desktop.Views;

public partial class OwnerSettingsView : UserControl
{
    private MainViewModel? _boundMainVm;

    public OwnerSettingsView()
    {
        InitializeComponent();

        RoiEditor.ZoneDrawn += norm =>
        {
            if (DataContext is MainViewModel vm)
                vm.CameraSetup.AddDrawnZone(norm);
            else if (RoiEditor.DataContext is CameraSetupViewModel camVm)
                camVm.AddDrawnZone(norm);
        };
        RoiEditor.BeforeZoneMutation += () =>
        {
            if (DataContext is MainViewModel vm)
                vm.CameraSetup.PushZoneUndoSnapshot();
            else if (RoiEditor.DataContext is CameraSetupViewModel camVm)
                camVm.PushZoneUndoSnapshot();
        };
        RoiEditor.ZoneModified += () =>
        {
            if (DataContext is MainViewModel vm)
                vm.CameraSetup.NotifyZoneEditedFromCanvas();
            else if (RoiEditor.DataContext is CameraSetupViewModel camVm)
                camVm.NotifyZoneEditedFromCanvas();
        };

        DataContextChanged += (_, e) =>
        {
            if (_boundMainVm is not null)
                _boundMainVm.ApexFlowEditor.ScrollToNodeRequested -= OnScrollToNodeRequested;
            _boundMainVm = e.NewValue as MainViewModel;
            if (_boundMainVm is not null)
                _boundMainVm.ApexFlowEditor.ScrollToNodeRequested += OnScrollToNodeRequested;
        };
    }

    public void SelectSubTab(int index)
    {
        if (index >= 0 && index < OwnerSubTabs.Items.Count)
            OwnerSubTabs.SelectedIndex = index;
    }

    private void OnScrollToNodeRequested(object? sender, string instanceId)
    {
        if (string.IsNullOrWhiteSpace(instanceId) || string.IsNullOrEmpty(ApexXmlBox.Text))
            return;

        var marker = $"instance_id=\"{instanceId}\"";
        int idx = ApexXmlBox.Text.IndexOf(marker, StringComparison.Ordinal);
        if (idx < 0) return;

        int lineStart = ApexXmlBox.Text.LastIndexOf('\n', idx);
        lineStart = lineStart < 0 ? 0 : lineStart + 1;
        int lineEnd = ApexXmlBox.Text.IndexOf('\n', idx);
        int len = (lineEnd < 0 ? ApexXmlBox.Text.Length : lineEnd) - lineStart;

        int lineIdx = ApexXmlBox.GetLineIndexFromCharacterIndex(lineStart);
        if (lineIdx >= 0)
            ApexXmlBox.ScrollToLine(lineIdx);
        ApexXmlBox.Select(lineStart, Math.Max(0, len));
    }

    private void UserControl_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (OwnerSubTabs.SelectedIndex != 0 || _boundMainVm is null)
            return;
        if (Keyboard.Modifiers == ModifierKeys.Control && e.Key == Key.Z && Keyboard.FocusedElement is not TextBoxBase)
        {
            if (_boundMainVm.CameraSetup.UndoZoneEditCommand.CanExecute(null))
            {
                _boundMainVm.CameraSetup.UndoZoneEditCommand.Execute(null);
                e.Handled = true;
            }
        }
    }
}
