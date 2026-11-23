using System.Windows;
using StoreGuard.Desktop.ViewModels;

namespace StoreGuard.Desktop.Views;

public partial class MainWindow : Window
{
    private MainViewModel? _boundVm;

    public MainWindow()
    {
        InitializeComponent();
        DataContextChanged += (_, e) =>
        {
            if (_boundVm is not null)
                _boundVm.RequestAdminSubTab -= OnRequestAdminSubTab;
            _boundVm = e.NewValue as MainViewModel;
            if (_boundVm is not null)
                _boundVm.RequestAdminSubTab += OnRequestAdminSubTab;
        };
    }

    private void OnRequestAdminSubTab(int subTabIndex)
        => OwnerSettingsPanel.SelectSubTab(subTabIndex);
}
