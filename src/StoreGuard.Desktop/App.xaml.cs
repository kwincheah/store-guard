using System.IO;
using System.Windows;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using StoreGuard.Core.Interfaces;
using StoreGuard.Desktop.ViewModels;
using StoreGuard.Desktop.Views;
using StoreGuard.Infrastructure.Engine;
using StoreGuard.Infrastructure.Notifications;
using StoreGuard.Infrastructure.Persistence;
using StoreGuard.Infrastructure.Sdk;
using StoreGuard.Infrastructure.Security;
using StoreGuard.Infrastructure.Services;

namespace StoreGuard.Desktop;

public partial class App : Application
{
    private ServiceProvider? _sp;
    private MainViewModel? _mainVm;
    private AppFileLogger? _appLogger;
    private readonly CancellationTokenSource _appCts = new();

    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        var rootDir = ResolveStoreGuardRoot();
        var runtimeDir = Path.Combine(rootDir, "runtime_data");
        Directory.CreateDirectory(runtimeDir);
        var logDir = Path.Combine(runtimeDir, "logs");
        _appLogger = new AppFileLogger(logDir);
        _appLogger.Info("StoreGuard Desktop starting...");

        DispatcherUnhandledException += (_, exArgs) =>
        {
            _appLogger?.Error("Unhandled WPF Dispatcher exception", exArgs.Exception);
            MessageBox.Show(
                $"An unexpected UI error occurred (logged to desktop_app.log):\n{exArgs.Exception.Message}",
                "StoreGuard",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
            exArgs.Handled = true;
        };
        AppDomain.CurrentDomain.UnhandledException += (_, exArgs) =>
        {
            if (exArgs.ExceptionObject is Exception ex)
                _appLogger?.Error("AppDomain unhandled exception", ex);
        };
        TaskScheduler.UnobservedTaskException += (_, exArgs) =>
        {
            _appLogger?.Error("Unobserved background task exception", exArgs.Exception);
            exArgs.SetObserved();
        };

        var dbPath = Path.Combine(runtimeDir, "storeguard.db");
        var cfgPath = Path.Combine(runtimeDir, "cameras.json");
        var solPath = Path.Combine(runtimeDir, "storeguard.apex");

        var services = new ServiceCollection();
        services.AddSingleton(_appLogger);
        services.AddDbContextFactory<AppDbContext>(opt => opt.UseSqlite($"Data Source={dbPath}"));
        services.AddSingleton<ISecretProtector, DpapiSecretProtector>();
        services.AddSingleton<IEventRepository, EventRepository>();
        services.AddSingleton<CamerasJsonWriter>();
        services.AddSingleton(_ => new TheftEngineClient("http://127.0.0.1:8700"));
        services.AddSingleton<IEngineClient>(sp => sp.GetRequiredService<TheftEngineClient>());
        services.AddSingleton<IApexSolutionClient>(sp => sp.GetRequiredService<TheftEngineClient>());
        services.AddSingleton<ApexSolution>();
        services.AddSingleton<INotifier, TelegramNotifier>();
        services.AddSingleton(_ => new SoundAlertService());
        services.AddSingleton(_ => new PythonProcessHost(rootDir, cfgPath, port: 8700, solutionPath: solPath));
        MainWindow? windowRef = null;
        bool ShowConfirmDialog(string title, string message, string confirmText, bool isDestructive)
        {
            if (windowRef is null) return true;
            return windowRef.Dispatcher.Invoke(() =>
            {
                var dlg = new ConfirmDialog(title, message, confirmText, isDestructive) { Owner = windowRef };
                return dlg.ShowDialog() == true;
            });
        }

        services.AddSingleton(sp => new MainViewModel(
            sp.GetRequiredService<IEventRepository>(),
            sp.GetRequiredService<ISecretProtector>(),
            sp.GetRequiredService<IEngineClient>(),
            sp.GetRequiredService<INotifier>(),
            sp.GetRequiredService<CamerasJsonWriter>(),
            sp.GetRequiredService<PythonProcessHost>(),
            sp.GetRequiredService<SoundAlertService>(),
            cfgPath,
            sp.GetRequiredService<AppFileLogger>(),
            ShowConfirmDialog,
            sp.GetRequiredService<ApexSolution>(),
            solPath));
        services.AddSingleton<MainWindow>();

        _sp = services.BuildServiceProvider();
        _mainVm = _sp.GetRequiredService<MainWindow>() is var _ ? _sp.GetRequiredService<MainViewModel>() : null;
        var window = _sp.GetRequiredService<MainWindow>();
        windowRef = window;
        window.DataContext = _mainVm;
        _mainVm!.RequestOwnerUnlockAsync += (verifyAsync, showHint) =>
        {
            var dlg = new OwnerPinDialog(verifyAsync, showHint) { Owner = window };
            return Task.FromResult(dlg.ShowDialog() == true);
        };
        _mainVm.RequestOwnerPinAsync += () =>
        {
            var dlg = new OwnerPinDialog { Owner = window };
            return Task.FromResult(dlg.ShowDialog() == true ? dlg.EnteredPin : null);
        };
        window.Show();

        try
        {
            await _mainVm.InitializeAsync(_appCts.Token);
            _appLogger.Info("StoreGuard Desktop initialized cleanly.");
        }
        catch (Exception ex)
        {
            _appLogger.Error("StoreGuard startup error", ex);
            _mainVm.EngineBanner = $"Startup error: {ex.Message}";
        }
    }

    protected override void OnExit(ExitEventArgs e)
    {
        _appLogger?.Info("StoreGuard Desktop exiting...");
        _appCts.Cancel();
        if (_mainVm is not null)
        {
            try { Task.Run(() => _mainVm.ShutdownAsync()).Wait(TimeSpan.FromSeconds(3)); } catch { }
        }
        try { _sp?.GetService<PythonProcessHost>()?.StopSync(); } catch { }
        if (_sp is not null)
        {
            try { Task.Run(() => _sp.DisposeAsync().AsTask()).Wait(TimeSpan.FromSeconds(2)); } catch { }
        }
        base.OnExit(e);
    }

    private static string ResolveStoreGuardRoot()
    {
        var cur = new DirectoryInfo(AppContext.BaseDirectory);
        while (cur is not null)
        {
            if (Directory.Exists(Path.Combine(cur.FullName, "ai_engine")))
                return cur.FullName;
            cur = cur.Parent;
        }
        return AppContext.BaseDirectory;
    }
}
