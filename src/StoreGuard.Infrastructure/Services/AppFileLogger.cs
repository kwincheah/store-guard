using System.IO;
using System.Text;

namespace StoreGuard.Infrastructure.Services;

/// <summary>Thread-safe rotating file logger writing desktop diagnostics to runtime_data/logs/desktop_app.log.</summary>
public sealed class AppFileLogger
{
    private const long MaxBytes = 10 * 1024 * 1024;
    private readonly string _logPath;
    private readonly object _gate = new();

    public string LogPath => _logPath;

    public AppFileLogger(string logDir)
    {
        Directory.CreateDirectory(logDir);
        _logPath = Path.Combine(logDir, "desktop_app.log");
    }

    public void Info(string message) => Write("INFO", message, null);

    public void Warn(string message) => Write("WARN", message, null);

    public void Error(string message, Exception? ex = null) => Write("ERROR", message, ex);

    private void Write(string level, string message, Exception? ex)
    {
        var sb = new StringBuilder();
        sb.Append(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"))
          .Append(" [").Append(level).Append("] ")
          .AppendLine(message);
        if (ex is not null)
            sb.AppendLine(ex.ToString());

        lock (_gate)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_logPath)!);
                if (File.Exists(_logPath) && new FileInfo(_logPath).Length >= MaxBytes)
                {
                    var backup = _logPath + ".1";
                    File.Move(_logPath, backup, overwrite: true);
                }
                File.AppendAllText(_logPath, sb.ToString(), Encoding.UTF8);
            }
            catch
            {
                // Avoid secondary crashes if disk/log handle is temporarily unavailable
            }
        }
    }
}
