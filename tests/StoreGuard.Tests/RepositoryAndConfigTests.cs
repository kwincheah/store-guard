using Microsoft.EntityFrameworkCore;
using StoreGuard.Core.Models;
using StoreGuard.Infrastructure.Engine;
using StoreGuard.Infrastructure.Persistence;
using StoreGuard.Infrastructure.Security;
using Xunit;

namespace StoreGuard.Tests;

internal sealed class TestDbFactory : IDbContextFactory<AppDbContext>
{
    private readonly DbContextOptions<AppDbContext> _options;

    public TestDbFactory(string dbPath)
    {
        _options = new DbContextOptionsBuilder<AppDbContext>()
            .UseSqlite($"Data Source={dbPath}")
            .Options;
    }

    public AppDbContext CreateDbContext() => new(_options);
}

public sealed class RepositoryAndConfigTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _dbPath;

    public RepositoryAndConfigTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sg_db_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
        _dbPath = Path.Combine(_tempDir, "test.db");
    }

    [Fact]
    public async Task Repository_InitializesDefaults_SavesCamerasEventsAndReviews()
    {
        var repo = new EventRepository(new TestDbFactory(_dbPath));
        await repo.InitializeAsync();

        // Default Owner PIN = 1234 and default retention = 30d / 20GB
        Assert.True(await repo.VerifyOwnerPinAsync("1234"));
        Assert.True(await repo.IsUsingDefaultPinAsync());
        var settings = await repo.GetSettingsAsync();
        Assert.Equal(30, settings.RetentionDays);
        Assert.Equal(20.0, settings.MaxEvidenceGb);
        Assert.Equal(65.0, settings.AlertThresholdPct);
        Assert.Equal("dark", settings.Theme);
        Assert.Equal(3, settings.BagInferStride);

        settings.Theme = "light";
        settings.BagInferStride = 2;
        await repo.SaveSettingsAsync(settings);
        var updatedSettings = await repo.GetSettingsAsync();
        Assert.Equal("light", updatedSettings.Theme);
        Assert.Equal(2, updatedSettings.BagInferStride);

        await repo.UpdateOwnerPinAsync("8429");
        Assert.False(await repo.IsUsingDefaultPinAsync());
        Assert.True(await repo.VerifyOwnerPinAsync("8429"));

        var protector = new DpapiSecretProtector();
        await repo.SaveCameraAsync(new CameraConfig
        {
            Id = "cam_01",
            Name = "Aisle 1",
            EncryptedRtspUrl = protector.Protect("rtsp://admin:pw@192.168.1.50:554/unicast/c1/s1/live"),
            Zones =
            [
                new CameraZone { Name = "shelf_1", Role = "shelf", XNorm = 0.1, YNorm = 0.2, WNorm = 0.2, HNorm = 0.5 },
            ],
        });

        await repo.InsertEventAsync(new TheftAlertEvent
        {
            EventId = "evt_001",
            CameraId = "cam_01",
            CameraName = "Aisle 1",
            TrackId = 7,
            BehaviorCategory = "high_risk_concealment_attempt",
            ConfidencePct = 88.5,
            Behaviors = ["reach_and_tuck"],
            Justification = "Concealed item into bag",
            GridPath = "grid.jpg",
            ClipPath = "clip.mp4",
            TimestampUtc = DateTime.UtcNow,
        });

        await repo.SubmitReviewAsync("evt_001", "ConfirmedTheft", "Verified on aisle 1", "CounterStaff", null);
        var hist = await repo.QueryHistoryAsync(null, null, "cam_01", "ConfirmedTheft");
        Assert.Single(hist);
        Assert.Equal("CounterStaff", hist[0].Review?.ReviewedBy);
        Assert.Equal("ConfirmedTheft", hist[0].Review?.Verdict);

        var csv = EventRepository.BuildCsvReport(hist);
        Assert.Contains("evt_001", csv);
        Assert.Contains("ConfirmedTheft", csv);

        // CamerasJsonWriter writes plaintext credentials and bag_infer_stride to runtime cameras.json cleanly
        var cfgOut = Path.Combine(_tempDir, "cameras.json");
        var cams = await repo.GetCamerasAsync();
        new CamerasJsonWriter(protector).WriteConfigAtomic(cfgOut, updatedSettings, cams);
        var jsonText = await File.ReadAllTextAsync(cfgOut);
        Assert.Contains("rtsp://admin:pw@192.168.1.50:554/unicast/c1/s1/live", jsonText);
        Assert.Contains("\"bag_infer_stride\": 2", jsonText);

        // DeleteReviewAsync cleanly resets event back to Unreviewed
        await repo.DeleteReviewAsync("evt_001");
        var unrev = await repo.QueryHistoryAsync(null, null, "ALL", "Unreviewed");
        Assert.Contains(unrev, e => e.Id == "evt_001");
    }

    public void Dispose()
    {
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(_tempDir, true); } catch { }
    }
}
