using StoreGuard.Core.Models;

namespace StoreGuard.Core.Interfaces;

public interface IEngineClient : IAsyncDisposable
{
    event EventHandler<TheftAlertEvent>? OnTheftAlert;
    event EventHandler<CameraHealthEvent>? OnCameraHealth;
    event EventHandler<VlmErrorEvent>? OnVlmError;
    event EventHandler<EngineMetricsEvent>? OnEngineMetrics;
    event EventHandler<bool>? OnConnectionChanged;

    bool IsConnected { get; }
    Task StartListeningAsync(CancellationToken ct);
    Task<EngineHealthResponse?> GetHealthAsync(CancellationToken ct);
    Task<byte[]?> GetSnapshotAsync(string cameraId, bool overlays, CancellationToken ct, int maxWidth = 0, int quality = 85, string? overlayLayers = null);
    Task<bool> UpdateCameraZonesAsync(string cameraId, IReadOnlyList<CameraZoneDto> zones, CancellationToken ct);
    Task<bool> ReloadAllCamerasAsync(CancellationToken ct);
    Task<bool> RerunCameraDetectionAsync(string cameraId, CancellationToken ct);
    Task<DeviceUpdateResponse?> SetDeviceAsync(string device, CancellationToken ct);
    Task<bool> RequestShutdownAsync(CancellationToken ct);
}

public interface IApexSolutionClient
{
    Task<ApexSolutionInfo?> GetSolutionAsync(CancellationToken ct = default);
    Task<ApexSolutionXmlResponse?> GetSolutionXmlAsync(CancellationToken ct = default);
    Task<ApexSolutionInfo?> LoadSolutionAsync(string? path = null, string? xml = null, CancellationToken ct = default);
    Task<ApexSolutionSaveResponse?> SaveSolutionAsync(string? path = null, CancellationToken ct = default);
    Task<List<ApexProcedureSummary>> ListProceduresAsync(CancellationToken ct = default);
    Task<ApexProcedureSummary?> GetProcedureAsync(string name, CancellationToken ct = default);
    Task<ApexProcedureRunResult?> RunProcedureOnceAsync(string name, CancellationToken ct = default);
    Task<ApexProcedureSummary?> SetProcedureParamAsync(string name, string nodeId, string paramName, object? value, CancellationToken ct = default);
}

public interface INotifier
{
    Task<(bool Success, string Message)> SendTheftAlertAsync(TheftAlertEvent alert, CancellationToken ct);
    Task<(bool Success, string Message)> SendTestMessageAsync(string botToken, string chatId, CancellationToken ct);
}

public interface ISecretProtector
{
    string Protect(string plaintext);
    string Unprotect(string protectedText);
}

public interface IEventRepository
{
    Task InitializeAsync(CancellationToken ct = default);
    Task<AppSettingsModel> GetSettingsAsync(CancellationToken ct = default);
    Task SaveSettingsAsync(AppSettingsModel settings, CancellationToken ct = default);

    Task<List<CameraConfig>> GetCamerasAsync(CancellationToken ct = default);
    Task SaveCameraAsync(CameraConfig camera, CancellationToken ct = default);
    Task ReplaceCameraZonesAsync(string cameraId, List<CameraZone> zones, CancellationToken ct = default);
    Task DeleteCameraAsync(string cameraId, CancellationToken ct = default);

    Task InsertEventAsync(TheftAlertEvent alert, CancellationToken ct = default);
    Task<List<EventRecord>> GetRecentEventsAsync(int limit = 100, CancellationToken ct = default);
    Task<List<EventRecord>> QueryHistoryAsync(DateTime? fromUtc, DateTime? toUtc, string? cameraId, string? verdictFilter, CancellationToken ct = default);
    Task SubmitReviewAsync(string eventId, string verdict, string note, string reviewedBy = "CounterStaff", int? userId = null, CancellationToken ct = default);

    Task<bool> VerifyOwnerPinAsync(string pin, CancellationToken ct = default);
    Task<bool> IsUsingDefaultPinAsync(CancellationToken ct = default);
    Task UpdateOwnerPinAsync(string newPin, CancellationToken ct = default);
    Task DeleteReviewAsync(string eventId, CancellationToken ct = default);
}
