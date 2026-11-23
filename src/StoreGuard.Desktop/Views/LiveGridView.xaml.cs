using System.Windows;
using System.Windows.Controls;
using StoreGuard.Desktop.ViewModels;

namespace StoreGuard.Desktop.Views;

public partial class LiveGridView : UserControl
{
    public LiveGridView()
    {
        InitializeComponent();
        DataContextChanged += (_, _) => SyncViewportSize();
    }

    private void GridViewportHost_Loaded(object sender, RoutedEventArgs e) => SyncViewportSize();

    private void GridViewportHost_SizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (DataContext is LiveGridViewModel vm && e.NewSize.Width > 0 && e.NewSize.Height > 0)
            vm.UpdateViewportSize(e.NewSize.Width, e.NewSize.Height);
    }

    private void SyncViewportSize()
    {
        if (DataContext is LiveGridViewModel vm && GridViewportHost is not null && GridViewportHost.ActualWidth > 0 && GridViewportHost.ActualHeight > 0)
            vm.UpdateViewportSize(GridViewportHost.ActualWidth, GridViewportHost.ActualHeight);
    }
}
