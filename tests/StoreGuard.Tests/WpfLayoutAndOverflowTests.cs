using System.Net;
using System.Net.Http;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.EntityFrameworkCore;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Desktop.ViewModels;
using StoreGuard.Desktop.Views;
using StoreGuard.Infrastructure.Engine;
using StoreGuard.Infrastructure.Notifications;
using StoreGuard.Infrastructure.Persistence;
using StoreGuard.Infrastructure.Security;
using StoreGuard.Infrastructure.Services;
using Xunit;

namespace StoreGuard.Tests;

public sealed class WpfLayoutAndOverflowTests
{
    private static void EnsureWpfAppResources()
    {
        if (Application.Current is null)
            _ = new Application { ShutdownMode = ShutdownMode.OnExplicitShutdown };

        var res = Application.Current!.Resources;
        if (!res.Contains("BoolToVis"))
        {
            res.MergedDictionaries.Clear();
            res.MergedDictionaries.Add(new ResourceDictionary
            {
                Source = new Uri("pack://application:,,,/StoreGuard.Desktop;component/Styles/DesignTokens.xaml", UriKind.Absolute),
            });
            res.MergedDictionaries.Add(new ResourceDictionary
            {
                Source = new Uri("pack://application:,,,/StoreGuard.Desktop;component/Styles/Controls.xaml", UriKind.Absolute),
            });
            res["BoolToVis"] = new Desktop.Converters.BoolToVisibilityConverter();
            res["InvertBoolToVis"] = new Desktop.Converters.BoolToVisibilityConverter { Invert = true };
            res["NotNullToVis"] = new Desktop.Converters.NullToVisibilityConverter();
            res["NullToVis"] = new Desktop.Converters.NullToVisibilityConverter { Invert = true };
            res["StrToVis"] = new Desktop.Converters.StringNotEmptyToVisibilityConverter();
            res["PathToBmp"] = new Desktop.Converters.PathToBitmapConverter();
            res["BytesToBmp"] = new Desktop.Converters.BytesToBitmapConverter();
            res["HealthToBrush"] = new Desktop.Converters.HealthStateToBrushConverter();
            res["VerdictToBrush"] = new Desktop.Converters.VerdictToBrushConverter();
            res["VerdictToTintBrush"] = new Desktop.Converters.VerdictToTintBrushConverter();
            res["VerdictToTextBrush"] = new Desktop.Converters.VerdictToTextBrushConverter();
            res["VerdictToBorderBrush"] = new Desktop.Converters.VerdictToBorderBrushConverter();
            res["VerdictToAccentBrush"] = new Desktop.Converters.VerdictToAccentBrushConverter();
        }
    }

    private static IEnumerable<FrameworkElement> Descendants(DependencyObject root)
    {
        int count = VisualTreeHelper.GetChildrenCount(root);
        for (int i = 0; i < count; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is FrameworkElement fe)
                yield return fe;
            foreach (var sub in Descendants(child))
                yield return sub;
        }
    }

    private static void AssertNoChildOverflow(Window win, FrameworkElement root, double width, double height)
    {
        win.Width = width;
        win.Height = height;
        win.UpdateLayout();

        // Pump dispatcher once so DataTriggers and VisualTrees settle after tab/size changes
        var frame = new System.Windows.Threading.DispatcherFrame();
        System.Windows.Threading.Dispatcher.CurrentDispatcher.BeginInvoke(
            System.Windows.Threading.DispatcherPriority.Render,
            new Action(() => frame.Continue = false));
        System.Windows.Threading.Dispatcher.PushFrame(frame);
        win.UpdateLayout();

        Assert.True(root.ActualWidth <= width + 1.0 && root.ActualWidth >= width - 40.0, $"Unexpected root width {root.ActualWidth}");

        // Every actually-visible element outside a horizontal ScrollViewer must stay within window bounds
        foreach (var fe in Descendants(root))
        {
            if (!fe.IsVisible || fe.ActualWidth <= 0 || fe.ActualHeight <= 0)
                continue;

            bool insideHorizScroll = false;
            for (DependencyObject? p = VisualTreeHelper.GetParent(fe); p is not null; p = VisualTreeHelper.GetParent(p))
            {
                if (p is ScrollViewer sv && sv.HorizontalScrollBarVisibility != ScrollBarVisibility.Disabled)
                {
                    insideHorizScroll = true;
                    break;
                }
            }
            if (insideHorizScroll)
                continue;

            var topLeft = fe.TranslatePoint(new Point(0, 0), root);
            var bottomRight = fe.TranslatePoint(new Point(fe.ActualWidth, fe.ActualHeight), root);
            var chain = new List<string>();
            for (DependencyObject? p = fe; p is not null && chain.Count < 6; p = VisualTreeHelper.GetParent(p))
                chain.Add(p is TextBlock tb ? $"TextBlock('{tb.Text[..Math.Min(25, tb.Text.Length)]}')" : p.GetType().Name);
            string desc = $"{string.Join("<-", chain)}(dc='{fe.DataContext?.GetType().Name}',tab={((fe.DataContext as MainViewModel)?.SelectedTabIndex.ToString() ?? "?")})";
            Assert.True(topLeft.X >= -2.0, $"{desc} overflowed left edge at ({topLeft.X:F1}) for {width}x{height}");
            Assert.True(topLeft.Y >= -2.0, $"{desc} overflowed top edge at ({topLeft.Y:F1}) for {width}x{height}");
            Assert.True(bottomRight.X <= root.ActualWidth + 2.5,
                $"{desc} overflowed right edge: {bottomRight.X:F1} > {root.ActualWidth:F1} at {width}x{height}");

            // Check sibling overlap inside non-Grid panels when siblings are visible
            if (fe is Panel panel && panel is StackPanel or WrapPanel)
            {
                var visibleChildren = panel.Children.OfType<FrameworkElement>()
                    .Where(c => c.IsVisible && c.ActualWidth > 0 && c.ActualHeight > 0)
                    .ToList();
                for (int i = 0; i < visibleChildren.Count; i++)
                {
                    var r1 = new Rect(visibleChildren[i].TranslatePoint(new Point(0, 0), root), visibleChildren[i].RenderSize);
                    for (int j = i + 1; j < visibleChildren.Count; j++)
                    {
                        var r2 = new Rect(visibleChildren[j].TranslatePoint(new Point(0, 0), root), visibleChildren[j].RenderSize);
                        r1.Intersect(r2);
                        Assert.True(r1.IsEmpty || r1.Width < 1.5 || r1.Height < 1.5,
                            $"Sibling overlap inside {panel.GetType().Name} at {width}x{height}: {visibleChildren[i].GetType().Name} vs {visibleChildren[j].GetType().Name}");
                        r1 = new Rect(visibleChildren[i].TranslatePoint(new Point(0, 0), root), visibleChildren[i].RenderSize);
                    }
                }
            }
        }
    }

    [Fact]
    public void MainWindow_AllTabsAndSubTabs_LayoutCleanlyWithoutOverflow()
    {
        Exception? threadEx = null;
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_ui_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        var t = new Thread(() =>
        {
            try
            {
                EnsureWpfAppResources();
                var dbPath = Path.Combine(tempDir, "ui.db");
                var repo = new EventRepository(new TestDbFactory(dbPath));
                repo.InitializeAsync().GetAwaiter().GetResult();
                var protector = new DpapiSecretProtector();

                for (int i = 1; i <= 5; i++)
                {
                    repo.SaveCameraAsync(new CameraConfig
                    {
                        Id = $"cam_{i:D2}",
                        Name = $"Aisle {i} - Very Long Descriptive Camera Location Label",
                        EncryptedRtspUrl = protector.Protect($"rtsp://admin:pass@192.168.1.50:554/unicast/c{i}/s1/live"),
                        Enabled = true,
                        Zones = [new CameraZone { Name = "shelf_long_name_zone", Role = "shelf", XNorm = 0.1, YNorm = 0.2, WNorm = 0.3, HNorm = 0.4 }],
                    }).GetAwaiter().GetResult();
                }

                var stubHandler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"requested_device":"cuda:0","resolved_device":"cpu","using_gpu":false,"cuda_available":false,"fallback_note":"Running CPU fallback"}""", Encoding.UTF8, "application/json"),
                });
                var client = new TheftEngineClient("http://127.0.0.1:8799", stubHandler);
                var notifier = new TelegramNotifier(repo, protector, new HttpClient(stubHandler));
                var writer = new CamerasJsonWriter(protector);
                var cfgPath = Path.Combine(tempDir, "cameras.json");
                var host = new PythonProcessHost(tempDir, cfgPath, 8799);
                var vm = new MainViewModel(repo, protector, client, notifier, writer, host, new SoundAlertService(), cfgPath);

                vm.AlertFeed.LoadInitialAsync().GetAwaiter().GetResult();
                vm.CameraSetup.LoadAsync().GetAwaiter().GetResult();
                vm.Settings.LoadAsync().GetAwaiter().GetResult();
                vm.History.RefreshAsync().GetAwaiter().GetResult();
                vm.AlertFeed.AddIncomingAlert(new TheftAlertEvent
                {
                    EventId = "evt_ui_1",
                    CameraId = "cam_01",
                    CameraName = "Aisle 1 - Very Long Descriptive Camera Location Label",
                    TrackId = 12,
                    BehaviorCategory = "high_risk_concealment_attempt",
                    ConfidencePct = 89.4,
                    Behaviors = ["reach_and_tuck", "blading"],
                    Justification = new string('A', 220),
                });
                vm.SelectDeviceCommand.ExecuteAsync("cuda:0").GetAwaiter().GetResult();
                vm.IsOwnerUnlocked = true;

                var win = new MainWindow { DataContext = vm, ShowInTaskbar = false, WindowStyle = WindowStyle.None };
                win.Show();
                try
                {
                    var contentRoot = (FrameworkElement)win.Content;
                    foreach (var theme in new[] { "dark", "light" })
                    {
                        Desktop.Services.ThemeManager.ApplyTheme(theme);
                        Assert.Equal(theme, Desktop.Services.ThemeManager.CurrentTheme);
                        foreach (var (w, h) in new[] { (1120.0, 680.0), (1400.0, 860.0) })
                        {
                            for (int tab = 0; tab <= 3; tab++)
                            {
                                vm.SelectedTabIndex = tab;
                                if (tab == 1)
                                {
                                    vm.LiveGrid.StartAsync().GetAwaiter().GetResult();
                                    vm.LiveGrid.ApplyHealthEvent(new CameraHealthEvent { CameraId = "cam_01", State = "ONLINE", Fps = 8.0, IsLocalFile = true, ClipDetectionDone = true });
                                }
                                AssertNoChildOverflow(win, contentRoot, w, h);

                                if (tab == 3)
                                {
                                    var ownerView = Descendants(win).OfType<OwnerSettingsView>().First();
                                    var subTabs = Descendants(ownerView).OfType<TabControl>().First();
                                    subTabs.SelectedIndex = 0;
                                    AssertNoChildOverflow(win, contentRoot, w, h);
                                    subTabs.SelectedIndex = 1;
                                    AssertNoChildOverflow(win, contentRoot, w, h);
                                    if (string.IsNullOrWhiteSpace(vm.ApexFlowEditor.XmlContent))
                                        vm.ApexFlowEditor.TryLoadDefault();
                                    subTabs.SelectedIndex = 2;
                                    AssertNoChildOverflow(win, contentRoot, w, h);
                                }
                            }
                        }
                    }
                }
                finally
                {
                    Desktop.Services.ThemeManager.ApplyTheme("dark");
                    vm.LiveGrid.Stop();
                    win.Close();
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.InvokeShutdown();
                }
            }
            catch (Exception ex)
            {
                threadEx = ex;
            }
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();

        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(tempDir, true); } catch { }
        if (threadEx is not null)
            throw new Exception("WPF layout test failed: " + threadEx, threadEx);
    }

    [Fact]
    public void AdminUiRenaming_AndRoiCanvasHitTesting_WorkWithoutOwnerText()
    {
        Exception? threadEx = null;
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_admin_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        var t = new Thread(() =>
        {
            try
            {
                EnsureWpfAppResources();
                var dbPath = Path.Combine(tempDir, "admin.db");
                var repo = new EventRepository(new TestDbFactory(dbPath));
                repo.InitializeAsync().GetAwaiter().GetResult();
                Assert.True(repo.VerifyOwnerPinAsync("1234").GetAwaiter().GetResult());

                var protector = new DpapiSecretProtector();
                var stubHandler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.NotFound));
                var client = new TheftEngineClient("http://127.0.0.1:8798", stubHandler);
                var notifier = new TelegramNotifier(repo, protector, new HttpClient(stubHandler));
                var writer = new CamerasJsonWriter(protector);
                var cfgPath = Path.Combine(tempDir, "cameras.json");
                var host = new PythonProcessHost(tempDir, cfgPath, 8798);
                var vm = new MainViewModel(repo, protector, client, notifier, writer, host, new SoundAlertService(), cfgPath);
                vm.IsOwnerUnlocked = true;

                var win = new MainWindow { DataContext = vm, ShowInTaskbar = false, WindowStyle = WindowStyle.None };
                var dlg = new OwnerPinDialog();
                win.Show();
                try
                {
                    Assert.DoesNotContain("Owner", dlg.Title, StringComparison.OrdinalIgnoreCase);
                    foreach (var tb in Descendants(dlg).OfType<TextBlock>())
                        Assert.DoesNotContain("Owner", tb.Text ?? string.Empty, StringComparison.OrdinalIgnoreCase);

                    for (int tab = 0; tab <= 3; tab++)
                    {
                        vm.SelectedTabIndex = tab;
                        win.UpdateLayout();
                        var frame = new System.Windows.Threading.DispatcherFrame();
                        System.Windows.Threading.Dispatcher.CurrentDispatcher.BeginInvoke(
                            System.Windows.Threading.DispatcherPriority.Render,
                            new Action(() => frame.Continue = false));
                        System.Windows.Threading.Dispatcher.PushFrame(frame);
                        win.UpdateLayout();

                        if (tab == 3)
                        {
                            var ownerView = Descendants(win).OfType<OwnerSettingsView>().First();
                            var subTabs = Descendants(ownerView).OfType<TabControl>().First();
                            for (int st = 0; st <= 2; st++)
                            {
                                subTabs.SelectedIndex = st;
                                win.UpdateLayout();
                                foreach (var tb in Descendants(win).OfType<TextBlock>().Where(x => x.IsVisible))
                                    Assert.DoesNotContain("Owner", tb.Text ?? string.Empty, StringComparison.OrdinalIgnoreCase);
                                foreach (var btn in Descendants(win).OfType<Button>().Where(x => x.IsVisible))
                                {
                                    if (btn.Content is string s)
                                        Assert.DoesNotContain("Owner", s, StringComparison.OrdinalIgnoreCase);
                                }
                            }
                            subTabs.SelectedIndex = 0;
                            win.UpdateLayout();
                            var roiEditor = Descendants(ownerView).OfType<RoiCanvasEditor>().First();
                            var surface = Assert.IsType<Grid>(roiEditor.FindName("ImageSurface"));
                            Assert.NotNull(surface.Background);
                            Assert.Equal(System.Windows.Input.Cursors.Cross, surface.Cursor);
                        }
                        else
                        {
                            foreach (var tb in Descendants(win).OfType<TextBlock>().Where(x => x.IsVisible))
                                Assert.DoesNotContain("Owner", tb.Text ?? string.Empty, StringComparison.OrdinalIgnoreCase);
                        }
                    }
                }
                finally
                {
                    dlg.Close();
                    win.Close();
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.InvokeShutdown();
                }
            }
            catch (Exception ex)
            {
                threadEx = ex;
            }
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();

        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(tempDir, true); } catch { }
        if (threadEx is not null)
            throw new Exception("Admin UI / ROI test failed: " + threadEx, threadEx);
    }

    [Fact]
    public async Task CameraSetupViewModel_HotReloadZones_PreservesSelectedCameraSnapshotAndSyncsJson()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_cam_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "cam.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            var protector = new DpapiSecretProtector();

            await repo.SaveCameraAsync(new CameraConfig { Id = "cam_01", Name = "Aisle 1", Enabled = true });
            await repo.SaveCameraAsync(new CameraConfig { Id = "cam_02", Name = "Aisle 2", Enabled = true });

            var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
            await using var client = new TheftEngineClient("http://127.0.0.1:8797", handler);
            bool jsonSynced = false;

            var vm = new CameraSetupViewModel(
                repo,
                protector,
                client,
                () => Task.CompletedTask,
                () => { jsonSynced = true; return Task.CompletedTask; });

            await vm.LoadAsync();
            vm.SelectedCamera = vm.Cameras.First(c => c.Id == "cam_02");
            vm.SnapshotBytes = [10, 20, 30, 40];
            vm.AddDrawnZone([0.1, 0.2, 0.3, 0.4]);

            await vm.SaveZonesOnlyHotReloadCommand.ExecuteAsync(null);

            Assert.True(jsonSynced);
            Assert.NotNull(vm.SelectedCamera);
            Assert.Equal("cam_02", vm.SelectedCamera!.Id);
            Assert.Equal([10, 20, 30, 40], vm.SnapshotBytes);
            Assert.Single(vm.Zones);
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public void RoiCanvasEditor_MouseDragRelease_DoesNotCrash_AndDistinguishesShelfFromAisle()
    {
        Exception? threadEx = null;
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_roi_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        var t = new Thread(() =>
        {
            try
            {
                EnsureWpfAppResources();
                var dbPath = Path.Combine(tempDir, "roi.db");
                var repo = new EventRepository(new TestDbFactory(dbPath));
                repo.InitializeAsync().GetAwaiter().GetResult();
                var protector = new DpapiSecretProtector();
                var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.NotFound));
                var client = new TheftEngineClient("http://127.0.0.1:8796", handler);
                var camVm = new CameraSetupViewModel(repo, protector, client, () => Task.CompletedTask);

                Assert.True(camVm.IsDrawShelf);
                Assert.Equal("shelf_1", camVm.NewZoneName);
                camVm.SelectDrawRoleCommand.Execute("aisle");
                Assert.True(camVm.IsDrawAisle);
                Assert.Equal("aisle_1", camVm.NewZoneName);

                var editor = new RoiCanvasEditor { Width = 640, Height = 360, ActiveDrawRole = "shelf" };
                var win = new Window { Content = editor, Width = 680, Height = 420, ShowInTaskbar = false, WindowStyle = WindowStyle.None };
                win.Show();
                try
                {
                    win.UpdateLayout();
                    var surface = Assert.IsType<Grid>(editor.FindName("ImageSurface"));

                    // Real mouse-down -> move -> mouse-up sequence triggers CaptureMouse + ReleaseMouseCapture (LostMouseCapture)
                    var down = new System.Windows.Input.MouseButtonEventArgs(
                        System.Windows.Input.Mouse.PrimaryDevice, Environment.TickCount, System.Windows.Input.MouseButton.Left)
                    { RoutedEvent = UIElement.MouseLeftButtonDownEvent };
                    surface.RaiseEvent(down);

                    var move = new System.Windows.Input.MouseEventArgs(
                        System.Windows.Input.Mouse.PrimaryDevice, Environment.TickCount)
                    { RoutedEvent = UIElement.MouseMoveEvent };
                    surface.RaiseEvent(move);

                    var up = new System.Windows.Input.MouseButtonEventArgs(
                        System.Windows.Input.Mouse.PrimaryDevice, Environment.TickCount, System.Windows.Input.MouseButton.Left)
                    { RoutedEvent = UIElement.MouseLeftButtonUpEvent };
                    surface.RaiseEvent(up);

                    // Distinct Shelf (solid orange) vs Aisle (dashed sky-blue) rendering
                    camVm.AddDrawnZone([0.10, 0.20, 0.25, 0.40]);
                    camVm.SelectDrawRoleCommand.Execute("shelf");
                    camVm.AddDrawnZone([0.40, 0.25, 0.20, 0.35]);
                    editor.Zones = camVm.Zones;
                    win.UpdateLayout();

                    var overlay = Assert.IsType<Canvas>(editor.FindName("OverlayCanvas"));
                    var rects = overlay.Children.OfType<System.Windows.Shapes.Rectangle>().ToList();
                    Assert.Equal(2, rects.Count);
                    Assert.NotNull(rects[0].StrokeDashArray);
                    Assert.True(rects[0].StrokeDashArray.Count > 0);   // Aisle drawn dashed underneath
                    Assert.True(rects[1].StrokeDashArray is null || rects[1].StrokeDashArray.Count == 0); // Shelf solid on top
                }
                finally
                {
                    win.Close();
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.InvokeShutdown();
                }
            }
            catch (Exception ex)
            {
                threadEx = ex;
            }
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();

        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(tempDir, true); } catch { }
        if (threadEx is not null)
            throw new Exception("ROI drag/role test failed: " + threadEx, threadEx);
    }

    [Fact]
    public async Task HistoryViewModel_IncludesEventsCreatedToday_AndLoggerWritesDesktopLog()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_hist_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "hist.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            await repo.InsertEventAsync(new TheftAlertEvent
            {
                EventId = "evt_today_1",
                CameraId = "cam_01",
                CameraName = "Aisle 1",
                TrackId = 4,
                BehaviorCategory = "high_risk_concealment_attempt",
                ConfidencePct = 91.0,
                Behaviors = ["reach_and_tuck"],
                Justification = "Concealed bottle in bag",
                TimestampUtc = DateTime.UtcNow,
            });

            var histVm = new HistoryViewModel(repo);
            await histVm.RefreshAsync();
            Assert.Contains(histVm.Items, x => x.EventId == "evt_today_1");

            var logger = new AppFileLogger(Path.Combine(tempDir, "logs"));
            logger.Info("Desktop smoke check");
            Assert.True(File.Exists(logger.LogPath));
            Assert.Contains("Desktop smoke check", await File.ReadAllTextAsync(logger.LogPath));
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public void SecretInputs_AreMaskedByDefault_AndAutoHideWhenHidden()
    {
        Exception? threadEx = null;
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_sec_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        var t = new Thread(() =>
        {
            try
            {
                EnsureWpfAppResources();
                var dbPath = Path.Combine(tempDir, "sec.db");
                var repo = new EventRepository(new TestDbFactory(dbPath));
                repo.InitializeAsync().GetAwaiter().GetResult();
                var protector = new DpapiSecretProtector();
                var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
                var client = new TheftEngineClient("http://127.0.0.1:8794", handler);
                var notifier = new TelegramNotifier(repo, protector, new HttpClient(handler));
                var writer = new CamerasJsonWriter(protector);
                var cfgPath = Path.Combine(tempDir, "cameras.json");
                var host = new PythonProcessHost(tempDir, cfgPath, 8794);
                var vm = new MainViewModel(repo, protector, client, notifier, writer, host, new SoundAlertService(), cfgPath);
                vm.IsOwnerUnlocked = true;
                vm.Settings.VlmApiKey = "sk-test-secret";
                vm.Settings.TelegramBotToken = "12345:bot-secret";
                vm.Settings.NewOwnerPin = "5678";
                vm.CameraSetup.UnvPassword = "nvr-password";

                var win = new MainWindow { DataContext = vm, Width = 1400, Height = 860, ShowInTaskbar = false, WindowStyle = WindowStyle.None };
                win.Show();
                try
                {
                    vm.SelectedTabIndex = 3;
                    win.UpdateLayout();
                    var ownerView = Descendants(win).OfType<OwnerSettingsView>().First();
                    var subTabs = Descendants(ownerView).OfType<TabControl>().First();

                    subTabs.SelectedIndex = 0;
                    win.UpdateLayout();
                    var exp = Descendants(ownerView).OfType<Expander>().First();
                    exp.IsExpanded = true;
                    win.UpdateLayout();

                    var camSecrets = Descendants(ownerView).OfType<SecretInputBox>().Where(s => s.IsVisible).ToList();
                    Assert.Single(camSecrets);
                    Assert.False(camSecrets[0].IsRevealed);
                    Assert.Equal(Visibility.Visible, ((PasswordBox)camSecrets[0].FindName("SecretPasswordBox")).Visibility);
                    Assert.Equal(Visibility.Collapsed, ((TextBox)camSecrets[0].FindName("PlainTextBox")).Visibility);

                    subTabs.SelectedIndex = 1;
                    win.UpdateLayout();
                    var vlmSecrets = Descendants(ownerView).OfType<SecretInputBox>().Where(s => s.IsVisible).ToList();
                    Assert.Equal(3, vlmSecrets.Count);
                    foreach (var s in vlmSecrets)
                    {
                        Assert.False(s.IsRevealed);
                        Assert.Equal(Visibility.Visible, ((PasswordBox)s.FindName("SecretPasswordBox")).Visibility);
                        Assert.Equal(Visibility.Collapsed, ((TextBox)s.FindName("PlainTextBox")).Visibility);
                    }

                    vlmSecrets[0].IsRevealed = true;
                    win.UpdateLayout();
                    Assert.Equal(Visibility.Visible, ((TextBox)vlmSecrets[0].FindName("PlainTextBox")).Visibility);

                    // Switching away auto-hides revealed secrets
                    subTabs.SelectedIndex = 0;
                    win.UpdateLayout();
                    Assert.False(vlmSecrets[0].IsRevealed);
                }
                finally
                {
                    vm.LiveGrid.Stop();
                    win.Close();
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.InvokeShutdown();
                }
            }
            catch (Exception ex)
            {
                threadEx = ex;
            }
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();

        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(tempDir, true); } catch { }
        if (threadEx is not null)
            throw new Exception("SecretInput test failed: " + threadEx, threadEx);
    }

    [Fact]
    public void OwnerPinDialog_KeyboardContracts_AndInlineErrorVerification()
    {
        Exception? threadEx = null;
        var t = new Thread(() =>
        {
            try
            {
                EnsureWpfAppResources();
                var dlg = new OwnerPinDialog(pin => Task.FromResult(pin == "9999"), showDefaultPinHint: false)
                {
                    ShowInTaskbar = false,
                };
                dlg.Show();
                try
                {
                    var buttons = Descendants(dlg).OfType<Button>().ToList();
                    var cancel = buttons.First(b => string.Equals(b.Content?.ToString(), "Cancel", StringComparison.Ordinal));
                    var unlock = buttons.First(b => (b.Content?.ToString() ?? "").Contains("Unlock", StringComparison.Ordinal));
                    Assert.True(cancel.IsCancel);
                    Assert.True(unlock.IsDefault);

                    var hint = Assert.IsType<TextBlock>(dlg.FindName("DefaultPinHintText"));
                    Assert.Equal(Visibility.Collapsed, hint.Visibility);

                    var errorBanner = Assert.IsType<Border>(dlg.FindName("ErrorBanner"));
                    Assert.Equal(Visibility.Collapsed, errorBanner.Visibility);

                    dlg.ShowError("Incorrect Admin PIN. Try again.");
                    Assert.Equal(Visibility.Visible, errorBanner.Visibility);
                }
                finally
                {
                    dlg.Close();
                    System.Windows.Threading.Dispatcher.CurrentDispatcher.InvokeShutdown();
                }
            }
            catch (Exception ex)
            {
                threadEx = ex;
            }
        });
        t.SetApartmentState(ApartmentState.STA);
        t.Start();
        t.Join();
        if (threadEx is not null)
            throw new Exception("OwnerPinDialog test failed: " + threadEx, threadEx);
    }

    [Fact]
    public async Task AlertFeed_PlayClipDisabledWithoutFile_AndVerdictUndoRestoresState()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_undo_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "undo.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            await repo.InsertEventAsync(new TheftAlertEvent
            {
                EventId = "evt_undo_1",
                CameraId = "cam_01",
                CameraName = "Aisle 1",
                TrackId = 3,
                BehaviorCategory = "bag_concealment",
                ConfidencePct = 88.0,
                Behaviors = ["reach_and_tuck"],
                Justification = "Test",
                ClipPath = "",
                TimestampUtc = DateTime.UtcNow,
            });

            var feed = new AlertFeedViewModel(repo);
            await feed.LoadInitialAsync();
            Assert.NotNull(feed.SelectedAlert);
            Assert.False(feed.CanPlaySelectedClip);

            feed.PlayClipCommand.Execute(null);
            Assert.False(feed.ShowClipMode);

            await feed.ConfirmTheftCommand.ExecuteAsync(null);
            Assert.Equal("ConfirmedTheft", feed.SelectedAlert!.Verdict);
            Assert.True(feed.ShowUndoToast);
            Assert.Equal(0, feed.UnreviewedCount);

            await feed.UndoLastVerdictCommand.ExecuteAsync(null);
            Assert.Equal("Unreviewed", feed.SelectedAlert.Verdict);
            Assert.False(feed.ShowUndoToast);
            Assert.Equal(1, feed.UnreviewedCount);
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public async Task SettingsViewModel_ProviderSwitchSuggestsModel_AndTestVlmWorks()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_vlm_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "vlm.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            var protector = new DpapiSecretProtector();
            var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
            var notifier = new TelegramNotifier(repo, protector, new HttpClient(handler));

            var vm = new SettingsViewModel(repo, protector, notifier, () => Task.CompletedTask, null, () => handler);
            await vm.LoadAsync();
            Assert.True(vm.IsGeminiProvider);
            Assert.Equal("gemini-2.5-flash", vm.VlmModelName);
            Assert.False(vm.HasUnsavedChanges);

            vm.VlmProvider = "openai";
            Assert.False(vm.IsGeminiProvider);
            Assert.Equal("gpt-4o-mini", vm.VlmModelName);
            Assert.True(vm.HasUnsavedChanges);

            vm.VlmApiKey = "sk-test-probe";
            await vm.TestVlmConnectionCommand.ExecuteAsync(null);
            Assert.Contains("VLM Test OK", vm.StatusMessage);
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public async Task CameraSetupViewModel_ConfirmGuards_UnsavedChanges_AndUndoZoneEdit()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_guard_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "guard.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            var protector = new DpapiSecretProtector();
            await repo.SaveCameraAsync(new CameraConfig { Id = "cam_01", Name = "Aisle 1", Enabled = true });
            await repo.SaveCameraAsync(new CameraConfig { Id = "cam_02", Name = "Aisle 2", Enabled = true });

            var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
            await using var client = new TheftEngineClient("http://127.0.0.1:8793", handler);

            bool confirmResult = false;
            int confirmCalls = 0;
            var vm = new CameraSetupViewModel(
                repo,
                protector,
                client,
                () => Task.CompletedTask,
                null,
                null,
                (_, _, _, _) => { confirmCalls++; return confirmResult; });

            await vm.LoadAsync();
            Assert.False(vm.HasUnsavedChanges);

            // Add zone & undo
            vm.AddDrawnZone([0.1, 0.1, 0.2, 0.3]);
            Assert.Single(vm.Zones);
            Assert.True(vm.HasUnsavedChanges);
            Assert.True(vm.CanUndoZoneEdit);

            // Switch camera while dirty with confirmResult=false -> stays on cam_01 and keeps zone
            vm.SelectedCamera = vm.Cameras.First(c => c.Id == "cam_02");
            Assert.Equal(1, confirmCalls);
            Assert.Equal("cam_01", vm.SelectedCamera!.Id);
            Assert.Single(vm.Zones);

            vm.UndoZoneEditCommand.Execute(null);
            Assert.Empty(vm.Zones);

            // Delete blocked when confirmResult=false, allowed when confirmResult=true
            await vm.DeleteCameraCommand.ExecuteAsync(null);
            Assert.Equal(2, vm.Cameras.Count);

            confirmResult = true;
            await vm.DeleteCameraCommand.ExecuteAsync(null);
            Assert.Single(vm.Cameras);
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public async Task LiveGridViewModel_AdaptsLayoutAndAspectRatio_AndSupportsCheckboxesAndRerun()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_lg_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "lg.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            var protector = new DpapiSecretProtector();

            for (int i = 1; i <= 5; i++)
            {
                await repo.SaveCameraAsync(new CameraConfig
                {
                    Id = $"cam_{i:D2}",
                    Name = $"Aisle {i}",
                    EncryptedRtspUrl = protector.Protect($@"C:\videos\clip_{i}.mp4"),
                    Enabled = true,
                });
            }

            int rerunCalls = 0;
            var handler = new StubHttpHandler(req =>
            {
                if (req.RequestUri!.AbsolutePath == "/cameras/cam_01/rerun" && req.Method == HttpMethod.Post)
                {
                    rerunCalls++;
                    return new HttpResponseMessage(HttpStatusCode.OK);
                }
                return new HttpResponseMessage(HttpStatusCode.OK);
            });
            await using var client = new TheftEngineClient("http://127.0.0.1:8792", handler);

            var vm = new LiveGridViewModel(client, repo);
            await vm.StartAsync();
            try
            {
                vm.UpdateViewportSize(1100, 680);
                Assert.Equal(5, vm.AllTiles.Count);
                Assert.Equal(5, vm.VisibleTiles.Count);
                Assert.True(vm.IsLayoutAuto);
                Assert.Equal(3, vm.EffectiveColumns); // 5 cameras in Auto -> 3 cols

                // Uncheck 3 cameras -> 2 visible -> Auto adapts to 2 cols (larger boxes!)
                vm.AllTiles[2].IsSelected = false;
                vm.AllTiles[3].IsSelected = false;
                vm.AllTiles[4].IsSelected = false;
                Assert.Equal(2, vm.VisibleTiles.Count);
                Assert.Equal(2, vm.EffectiveColumns);

                // Uncheck 1 more -> 1 visible -> Auto adapts to 1 col filling viewport!
                double twoCamWidth = vm.VisibleTiles[0].TileMediaWidth;
                vm.AllTiles[1].IsSelected = false;
                Assert.Single(vm.VisibleTiles);
                Assert.Equal(1, vm.EffectiveColumns);
                Assert.True(vm.VisibleTiles[0].TileMediaWidth > twoCamWidth);

                // Dynamic aspect ratio adaptation (portrait 9:16 vs widescreen 16:9)
                Assert.True(vm.VisibleTiles[0].UpdateFeedDimensions(720, 1280));
                vm.RecalculateLayout();
                Assert.True(vm.VisibleTiles[0].TileMediaHeight > vm.VisibleTiles[0].TileMediaWidth);

                // Manual Clear & All checkboxes
                vm.ClearSelectedCamerasCommand.Execute(null);
                Assert.Empty(vm.VisibleTiles);
                Assert.True(vm.AllCamerasHidden);
                vm.SelectAllCamerasCommand.Execute(null);
                Assert.Equal(5, vm.VisibleTiles.Count);
                Assert.False(vm.AllCamerasHidden);

                // Rerun command clears ClipDetectionDone
                vm.ApplyHealthEvent(new CameraHealthEvent
                {
                    CameraId = "cam_01",
                    State = "ONLINE",
                    Fps = 12.0,
                    IsLocalFile = true,
                    ClipDetectionDone = true,
                });
                Assert.True(vm.AllTiles[0].IsLocalFile);
                Assert.True(vm.AllTiles[0].ClipDetectionDone);
                await vm.RerunCameraCommand.ExecuteAsync(vm.AllTiles[0]);
                Assert.Equal(1, rerunCalls);
                Assert.False(vm.AllTiles[0].ClipDetectionDone);
            }
            finally
            {
                vm.Stop();
            }
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public async Task LiveGridViewModel_PollsVisibleTilesInParallelWithSdProfileAndDecodesFrozenBitmap()
    {
        static byte[] MakeSolidJpeg(int width, int height)
        {
            var pixels = new byte[width * height * 3];
            var bmp = System.Windows.Media.Imaging.BitmapSource.Create(
                width,
                height,
                96,
                96,
                PixelFormats.Rgb24,
                null,
                pixels,
                width * 3);
            var enc = new System.Windows.Media.Imaging.JpegBitmapEncoder { QualityLevel = 75 };
            enc.Frames.Add(System.Windows.Media.Imaging.BitmapFrame.Create(bmp));
            using var ms = new MemoryStream();
            enc.Save(ms);
            return ms.ToArray();
        }

        byte[] sdJpg = MakeSolidJpeg(640, 360);
        byte[] hdJpg = MakeSolidJpeg(1280, 720);

        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_lg_par_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "lg_par.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            var protector = new DpapiSecretProtector();

            for (int i = 1; i <= 3; i++)
            {
                await repo.SaveCameraAsync(new CameraConfig
                {
                    Id = $"cam_{i:D2}",
                    Name = $"Aisle {i}",
                    EncryptedRtspUrl = protector.Protect($@"C:\videos\clip_{i}.mp4"),
                    Enabled = true,
                });
            }

            var queries = new System.Collections.Concurrent.ConcurrentBag<string>();
            var handler = new StubHttpHandler(req =>
            {
                var path = req.RequestUri!.AbsolutePath;
                if (path == "/health")
                {
                    return new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new StringContent(
                            """{"status":"ready","device":"cpu","using_gpu":false,"active_streams":3,"vlm_queue_depth":0,"cameras":[{"camera_id":"cam_01","state":"ONLINE","fps":8.0,"motion_active":false,"last_frame_age_ms":10,"error":"","is_local_file":false,"clip_detection_done":false,"loop_count":0},{"camera_id":"cam_02","state":"ONLINE","fps":8.0,"motion_active":false,"last_frame_age_ms":10,"error":"","is_local_file":false,"clip_detection_done":false,"loop_count":0},{"camera_id":"cam_03","state":"ONLINE","fps":8.0,"motion_active":false,"last_frame_age_ms":10,"error":"","is_local_file":false,"clip_detection_done":false,"loop_count":0}]}""",
                            Encoding.UTF8,
                            "application/json"),
                    };
                }
                if (path.EndsWith("/snapshot", StringComparison.Ordinal))
                {
                    var q = req.RequestUri.Query;
                    queries.Add(q);
                    var bytes = q.Contains("max_width=1280", StringComparison.Ordinal) ? hdJpg : sdJpg;
                    return new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new ByteArrayContent(bytes),
                    };
                }
                return new HttpResponseMessage(HttpStatusCode.NotFound);
            });
            await using var client = new TheftEngineClient("http://127.0.0.1:8791", handler);

            var vm = new LiveGridViewModel(client, repo);
            await vm.StartAsync();
            try
            {
                var deadline = DateTime.UtcNow.AddSeconds(3);
                while (DateTime.UtcNow < deadline && vm.VisibleTiles.Any(t => t.SnapshotImage is null))
                    await Task.Delay(25);

                Assert.All(vm.VisibleTiles, t =>
                {
                    Assert.True(t.HasSnapshot);
                    Assert.NotNull(t.SnapshotImage);
                    var bs = Assert.IsAssignableFrom<System.Windows.Media.Imaging.BitmapSource>(t.SnapshotImage);
                    Assert.True(bs.IsFrozen);
                    Assert.Equal(640, bs.PixelWidth);
                    Assert.Equal(360, bs.PixelHeight);
                });
                Assert.Contains(queries, q => q.Contains("max_width=640", StringComparison.Ordinal) && q.Contains("quality=75", StringComparison.Ordinal));

                vm.FocusTileCommand.Execute(vm.VisibleTiles[0]);
                var focusDeadline = DateTime.UtcNow.AddSeconds(3);
                while (DateTime.UtcNow < focusDeadline && !queries.Any(q => q.Contains("max_width=1280", StringComparison.Ordinal)))
                    await Task.Delay(25);
                Assert.Contains(queries, q => q.Contains("max_width=1280", StringComparison.Ordinal));
            }
            finally
            {
                vm.Stop();
            }
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }

    [Fact]
    public async Task LiveGridViewModel_Paginates20CamerasAndPollsOnlyActivePage()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"sg_lg_pg_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var dbPath = Path.Combine(tempDir, "lg_pg.db");
            var repo = new EventRepository(new TestDbFactory(dbPath));
            await repo.InitializeAsync();
            var protector = new DpapiSecretProtector();

            for (int i = 1; i <= 20; i++)
            {
                await repo.SaveCameraAsync(new CameraConfig
                {
                    Id = $"cam_{i:D2}",
                    Name = $"Aisle {i}",
                    EncryptedRtspUrl = protector.Protect($"rtsp://127.0.0.1/ch{i}"),
                    EncryptedSubStreamUrl = protector.Protect($"rtsp://127.0.0.1/ch{i}_sub"),
                    Enabled = true,
                });
            }

            var polledCameras = new System.Collections.Concurrent.ConcurrentBag<string>();
            var handler = new StubHttpHandler(req =>
            {
                var path = req.RequestUri!.AbsolutePath;
                if (path == "/health")
                {
                    return new HttpResponseMessage(HttpStatusCode.OK)
                    {
                        Content = new StringContent("""{"status":"ready","device":"cpu","using_gpu":false,"active_streams":20,"worker_threads":4,"cameras":[]}""", Encoding.UTF8, "application/json"),
                    };
                }
                if (path.EndsWith("/snapshot", StringComparison.OrdinalIgnoreCase))
                {
                    var parts = path.Split('/', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                        polledCameras.Add(parts[1]);
                    return new HttpResponseMessage(HttpStatusCode.ServiceUnavailable);
                }
                return new HttpResponseMessage(HttpStatusCode.NotFound);
            });
            await using var client = new TheftEngineClient("http://127.0.0.1:8792", handler);
            var vm = new LiveGridViewModel(client, repo);
            await vm.StartAsync();
            try
            {
                Assert.Equal(20, vm.AllTiles.Count);
                Assert.Equal(16, vm.VisibleTiles.Count);
                Assert.Equal(2, vm.TotalPages);
                Assert.Equal(1, vm.CurrentPage);
                Assert.True(vm.HasMultiplePages);
                Assert.False(vm.CanGoPreviousPage);
                Assert.True(vm.CanGoNextPage);

                await Task.Delay(120);
                var page1Polled = polledCameras.Distinct().ToHashSet(StringComparer.OrdinalIgnoreCase);
                Assert.All(page1Polled, id => Assert.Contains(id, Enumerable.Range(1, 16).Select(n => $"cam_{n:D2}")));

                vm.NextPageCommand.Execute(null);
                Assert.Equal(2, vm.CurrentPage);
                Assert.Equal(4, vm.VisibleTiles.Count);
                Assert.Equal("Page 2 of 2", vm.PageSummary);
                Assert.True(vm.CanGoPreviousPage);
                Assert.False(vm.CanGoNextPage);

                vm.SetGridLayoutCommand.Execute("2");
                Assert.Equal(1, vm.CurrentPage);
                Assert.Equal(5, vm.TotalPages);
                Assert.Equal(4, vm.VisibleTiles.Count);
            }
            finally
            {
                vm.Stop();
            }
        }
        finally
        {
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(tempDir, true); } catch { }
        }
    }
}
