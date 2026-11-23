using System.Diagnostics;
using System.Net.Http;

namespace StoreGuard.Infrastructure.Engine;

public sealed class PythonProcessHost : IAsyncDisposable
{
    private readonly string _workingDir;
    private readonly string _configPath;
    private readonly string _solutionPath;
    private readonly string _pythonExe;
    private readonly string _logDir;
    private readonly int _port;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(2) };
    private readonly object _gate = new();

    private Process? _process;
    private CancellationTokenSource? _watchdogCts;
    private Task? _watchdogTask;
    private StreamWriter? _logWriter;
    private volatile bool _stopping;

    public event Action<string>? OnStatusMessage;

    public PythonProcessHost(
        string workingDir,
        string configPath,
        int port = 8700,
        string? pythonExe = null,
        string? logDir = null,
        string? solutionPath = null)
    {
        _workingDir = workingDir;
        _configPath = configPath;
        _port = port;
        _pythonExe = pythonExe ?? ResolvePythonExecutable(workingDir);
        _logDir = logDir ?? Path.Combine(_workingDir, "runtime_data", "logs");
        _solutionPath = !string.IsNullOrWhiteSpace(solutionPath)
            ? solutionPath
            : Path.Combine(Path.GetDirectoryName(configPath) ?? _workingDir, "storeguard.apex");
    }

    public string SolutionPath => _solutionPath;

    public static string ResolvePythonExecutable(string workingDir)
    {
        var venvPy = Path.Combine(workingDir, ".venv", "Scripts", "python.exe");
        return File.Exists(venvPy) ? venvPy : "python";
    }

    public async Task StartAndMonitorAsync(CancellationToken ct = default)
    {
        _stopping = false;
        _watchdogCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        await EnsurePortClearedAsync(_watchdogCts.Token);
        if (_stopping || ct.IsCancellationRequested)
            return;
        LaunchProcess();
        _watchdogTask = Task.Run(() => MonitorLoopAsync(_watchdogCts.Token), _watchdogCts.Token);
    }

    private async Task EnsurePortClearedAsync(CancellationToken ct)
    {
        bool existingHealthy = false;
        try
        {
            using var probeCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            probeCts.CancelAfter(700);
            using var resp = await _http.GetAsync($"http://127.0.0.1:{_port}/health", probeCts.Token);
            existingHealthy = resp.IsSuccessStatusCode;
        }
        catch
        {
            return;
        }

        if (!existingHealthy)
            return;

        OnStatusMessage?.Invoke($"Stopping stale AI engine on port {_port}...");
        try
        {
            using var shutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            shutCts.CancelAfter(1200);
            await _http.PostAsync($"http://127.0.0.1:{_port}/shutdown", null, shutCts.Token);
        }
        catch
        {
            // Continue polling until port closes
        }

        var deadline = DateTime.UtcNow + TimeSpan.FromMilliseconds(2500);
        while (DateTime.UtcNow < deadline && !ct.IsCancellationRequested && !_stopping)
        {
            await Task.Delay(250, ct);
            try
            {
                using var checkCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
                checkCts.CancelAfter(400);
                using var check = await _http.GetAsync($"http://127.0.0.1:{_port}/health", checkCts.Token);
                if (!check.IsSuccessStatusCode)
                    return;
            }
            catch
            {
                return;
            }
        }
    }

    public async Task<bool> WaitForHealthyAsync(TimeSpan timeout, CancellationToken ct)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline && !ct.IsCancellationRequested)
        {
            try
            {
                using var resp = await _http.GetAsync($"http://127.0.0.1:{_port}/health", ct);
                if (resp.IsSuccessStatusCode)
                    return true;
            }
            catch
            {
                // Keep polling while engine loads ONNX models
            }
            await Task.Delay(500, ct);
        }
        return false;
    }

    private void LaunchProcess()
    {
        lock (_gate)
        {
            if (_process is { HasExited: false })
                return;

            Directory.CreateDirectory(_logDir);
            _logWriter?.Dispose();
            var fs = new FileStream(Path.Combine(_logDir, "ai_engine.log"), FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
            _logWriter = new StreamWriter(fs) { AutoFlush = true };

            var psi = new ProcessStartInfo
            {
                FileName = _pythonExe,
                Arguments = $"-m ai_engine.main --host 127.0.0.1 --port {_port} --config \"{_configPath}\" --solution \"{_solutionPath}\"",
                WorkingDirectory = _workingDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("APEX_VISION_PATH")))
            {
                var vmCloneDir = ResolveSiblingApexVisionDir(_workingDir);
                if (!string.IsNullOrWhiteSpace(vmCloneDir))
                    psi.Environment["APEX_VISION_PATH"] = vmCloneDir;
            }

            var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
            proc.OutputDataReceived += (_, e) => { if (e.Data is not null) SafeLog("[OUT] " + e.Data); };
            proc.ErrorDataReceived += (_, e) => { if (e.Data is not null) SafeLog("[ERR] " + e.Data); };
            proc.Start();
            proc.BeginOutputReadLine();
            proc.BeginErrorReadLine();
            _process = proc;
            OnStatusMessage?.Invoke($"AI engine started (PID {proc.Id})");
        }
    }

    private void SafeLog(string line)
    {
        lock (_gate)
        {
            try { _logWriter?.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss} {line}"); } catch { }
        }
    }

    private async Task MonitorLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested && !_stopping)
        {
            await Task.Delay(1500, ct);
            bool exited = false;
            lock (_gate)
            {
                exited = _process is null || _process.HasExited;
            }
            if (exited && !_stopping && !ct.IsCancellationRequested)
            {
                OnStatusMessage?.Invoke("AI engine exited unexpectedly; restarting in 3s...");
                await Task.Delay(3000, ct);
                if (!_stopping && !ct.IsCancellationRequested)
                    LaunchProcess();
            }
        }
    }

    public void StopSync()
    {
        _stopping = true;
        try { _watchdogCts?.Cancel(); } catch { }

        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(1));
            _http.PostAsync($"http://127.0.0.1:{_port}/shutdown", null, cts.Token).GetAwaiter().GetResult();
        }
        catch
        {
            // Fall through to tree kill
        }

        Process? proc;
        lock (_gate)
        {
            proc = _process;
            _process = null;
        }
        if (proc is not null)
        {
            try
            {
                if (!proc.HasExited && !proc.WaitForExit(1500))
                    proc.Kill(entireProcessTree: true);
            }
            catch
            {
                // Ignore if already exited
            }
            try { proc.Dispose(); } catch { }
        }

        lock (_gate)
        {
            try { _logWriter?.Dispose(); } catch { }
            _logWriter = null;
        }
    }

    public async ValueTask DisposeAsync()
    {
        _stopping = true;
        _watchdogCts?.Cancel();

        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            await _http.PostAsync($"http://127.0.0.1:{_port}/shutdown", null, cts.Token);
        }
        catch
        {
            // Fall through to tree kill
        }

        Process? proc;
        lock (_gate)
        {
            proc = _process;
            _process = null;
        }
        if (proc is not null)
        {
            try
            {
                if (!proc.HasExited && !proc.WaitForExit(2500))
                    proc.Kill(entireProcessTree: true);
            }
            catch
            {
                // Ignore if already exited
            }
            try { proc.Dispose(); } catch { }
        }

        if (_watchdogTask is not null)
        {
            try { await _watchdogTask; } catch { }
        }
        lock (_gate)
        {
            try { _logWriter?.Dispose(); } catch { }
            _logWriter = null;
        }
        _http.Dispose();
    }

    private static string? ResolveSiblingApexVisionDir(string workingDir)
    {
        try
        {
            var cur = new DirectoryInfo(Path.GetFullPath(workingDir));
            for (int i = 0; i < 6 && cur is not null; i++, cur = cur.Parent)
            {
                var cand = Path.Combine(cur.FullName, "VMClone");
                if (File.Exists(Path.Combine(cand, "apex_vision", "__init__.py")))
                    return cand;
            }
        }
        catch
        {
        }
        return null;
    }
}
