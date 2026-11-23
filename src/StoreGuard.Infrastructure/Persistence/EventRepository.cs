using System.Text;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;
using StoreGuard.Core.Utilities;

namespace StoreGuard.Infrastructure.Persistence;

public sealed class EventRepository : IEventRepository
{
    private readonly IDbContextFactory<AppDbContext> _factory;

    public EventRepository(IDbContextFactory<AppDbContext> factory)
    {
        _factory = factory;
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await db.Database.EnsureCreatedAsync(ct);
        try
        {
            await db.Database.ExecuteSqlRawAsync("PRAGMA journal_mode=WAL;", ct);
        }
        catch
        {
            // Ignore in-memory or non-WAL providers
        }

        try
        {
            var conn = db.Database.GetDbConnection();
            bool wasClosed = conn.State != System.Data.ConnectionState.Open;
            if (wasClosed)
                await conn.OpenAsync(ct);
            var cols = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            await using (var cmd = conn.CreateCommand())
            {
                cmd.CommandText = "PRAGMA table_info(SETTINGS);";
                await using var reader = await cmd.ExecuteReaderAsync(ct);
                while (await reader.ReadAsync(ct))
                {
                    if (reader.FieldCount > 1 && !reader.IsDBNull(1))
                        cols.Add(reader.GetString(1));
                }
            }
            if (!cols.Contains("PoseModel"))
                await db.Database.ExecuteSqlRawAsync("ALTER TABLE SETTINGS ADD COLUMN PoseModel TEXT NOT NULL DEFAULT 'yolo11n-pose.onnx';", ct);
            if (!cols.Contains("VlmUseNativeVideo"))
                await db.Database.ExecuteSqlRawAsync("ALTER TABLE SETTINGS ADD COLUMN VlmUseNativeVideo INTEGER NOT NULL DEFAULT 1;", ct);
            if (!cols.Contains("Theme"))
                await db.Database.ExecuteSqlRawAsync("ALTER TABLE SETTINGS ADD COLUMN Theme TEXT NOT NULL DEFAULT 'dark';", ct);
            if (!cols.Contains("BagInferStride"))
                await db.Database.ExecuteSqlRawAsync("ALTER TABLE SETTINGS ADD COLUMN BagInferStride INTEGER NOT NULL DEFAULT 3;", ct);

            var camCols = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            await using (var cmd = conn.CreateCommand())
            {
                cmd.CommandText = "PRAGMA table_info(CAMERAS);";
                await using var reader = await cmd.ExecuteReaderAsync(ct);
                while (await reader.ReadAsync(ct))
                {
                    if (reader.FieldCount > 1 && !reader.IsDBNull(1))
                        camCols.Add(reader.GetString(1));
                }
            }
            if (!camCols.Contains("EncryptedSubStreamUrl"))
                await db.Database.ExecuteSqlRawAsync("ALTER TABLE CAMERAS ADD COLUMN EncryptedSubStreamUrl TEXT NOT NULL DEFAULT '';", ct);
            if (wasClosed)
                await conn.CloseAsync();
        }
        catch
        {
            // Ignore in-memory or non-SQLite providers
        }

        var adminUser = await db.Users.FirstOrDefaultAsync(u => u.Username == "admin" || u.Username == "owner", ct);
        if (adminUser is null)
        {
            db.Users.Add(new UserAccount
            {
                Username = "admin",
                DisplayName = "Store Admin",
                Role = "Admin",
                PinHash = PinHasher.HashPin("1234"),
                IsActive = true,
            });
        }
        else if (adminUser.Username == "owner" && (!string.Equals(adminUser.DisplayName, "Store Admin", StringComparison.Ordinal) || !string.Equals(adminUser.Role, "Admin", StringComparison.Ordinal)))
        {
            adminUser.DisplayName = "Store Admin";
            adminUser.Role = "Admin";
        }

        if (!await db.Settings.AnyAsync(ct))
        {
            db.Settings.Add(new AppSettingsModel
            {
                Id = 1,
                Device = "auto",
                PoseModel = "yolo11n-pose.onnx",
                TargetFps = 8,
                ClipPreSec = 6.0,
                ClipPostSec = 4.0,
                RetentionDays = 30,
                MaxEvidenceGb = 20.0,
                VlmProvider = "gemini",
                VlmModelName = "gemini-2.5-flash",
                AlertThresholdPct = 65.0,
                VlmUseNativeVideo = true,
                Theme = "dark",
                BagInferStride = 3,
            });
        }

        await db.SaveChangesAsync(ct);
    }

    public async Task<AppSettingsModel> GetSettingsAsync(CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        return await db.Settings.AsNoTracking().FirstOrDefaultAsync(x => x.Id == 1, ct) ?? new AppSettingsModel();
    }

    public async Task SaveSettingsAsync(AppSettingsModel settings, CancellationToken ct = default)
    {
        settings.Id = 1;
        await using var db = await _factory.CreateDbContextAsync(ct);
        var existing = await db.Settings.FirstOrDefaultAsync(x => x.Id == 1, ct);
        if (existing is null)
            db.Settings.Add(settings);
        else
            db.Entry(existing).CurrentValues.SetValues(settings);
        await db.SaveChangesAsync(ct);
    }

    public async Task<List<CameraConfig>> GetCamerasAsync(CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        return await db.Cameras
            .AsNoTracking()
            .Include(c => c.Zones)
            .OrderBy(c => c.Id)
            .ToListAsync(ct);
    }

    public async Task SaveCameraAsync(CameraConfig camera, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var existing = await db.Cameras.Include(c => c.Zones).FirstOrDefaultAsync(c => c.Id == camera.Id, ct);
        if (existing is null)
        {
            foreach (var z in camera.Zones) z.CameraId = camera.Id;
            db.Cameras.Add(camera);
        }
        else
        {
            existing.Name = camera.Name;
            existing.EncryptedRtspUrl = camera.EncryptedRtspUrl;
            existing.EncryptedSubStreamUrl = camera.EncryptedSubStreamUrl;
            existing.Enabled = camera.Enabled;
            existing.DwellSec = camera.DwellSec;
            existing.ReleaseWaitSec = camera.ReleaseWaitSec;
            existing.CooldownSec = camera.CooldownSec;
            db.CameraZones.RemoveRange(existing.Zones);
            existing.Zones = camera.Zones.Select(z => new CameraZone
            {
                CameraId = camera.Id,
                Name = z.Name,
                Role = z.Role,
                XNorm = z.XNorm,
                YNorm = z.YNorm,
                WNorm = z.WNorm,
                HNorm = z.HNorm,
            }).ToList();
        }
        await db.SaveChangesAsync(ct);
    }

    public async Task ReplaceCameraZonesAsync(string cameraId, List<CameraZone> zones, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var existing = await db.CameraZones.Where(z => z.CameraId == cameraId).ToListAsync(ct);
        db.CameraZones.RemoveRange(existing);
        foreach (var z in zones)
        {
            db.CameraZones.Add(new CameraZone
            {
                CameraId = cameraId,
                Name = z.Name,
                Role = z.Role,
                XNorm = z.XNorm,
                YNorm = z.YNorm,
                WNorm = z.WNorm,
                HNorm = z.HNorm,
            });
        }
        await db.SaveChangesAsync(ct);
    }

    public async Task DeleteCameraAsync(string cameraId, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var existing = await db.Cameras.FirstOrDefaultAsync(c => c.Id == cameraId, ct);
        if (existing is not null)
        {
            db.Cameras.Remove(existing);
            await db.SaveChangesAsync(ct);
        }
    }

    public async Task InsertEventAsync(TheftAlertEvent alert, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        if (await db.Events.AnyAsync(e => e.Id == alert.EventId, ct))
            return;

        db.Events.Add(new EventRecord
        {
            Id = alert.EventId,
            CameraId = alert.CameraId,
            CameraName = alert.CameraName,
            TrackId = alert.TrackId,
            BehaviorCategory = alert.BehaviorCategory,
            ConfidencePct = alert.ConfidencePct,
            BehaviorsJson = JsonSerializer.Serialize(alert.Behaviors),
            Justification = alert.Justification,
            GridPath = alert.GridPath,
            ClipPath = alert.ClipPath,
            CreatedAtUtc = alert.TimestampUtc,
        });
        await db.SaveChangesAsync(ct);
    }

    public async Task<List<EventRecord>> GetRecentEventsAsync(int limit = 100, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        return await db.Events
            .AsNoTracking()
            .Include(e => e.Review)
            .OrderByDescending(e => e.CreatedAtUtc)
            .Take(Math.Max(1, limit))
            .ToListAsync(ct);
    }

    public async Task<List<EventRecord>> QueryHistoryAsync(
        DateTime? fromUtc,
        DateTime? toUtc,
        string? cameraId,
        string? verdictFilter,
        CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var q = db.Events.AsNoTracking().Include(e => e.Review).AsQueryable();

        if (fromUtc.HasValue) q = q.Where(e => e.CreatedAtUtc >= fromUtc.Value);
        if (toUtc.HasValue) q = q.Where(e => e.CreatedAtUtc <= toUtc.Value);
        if (!string.IsNullOrWhiteSpace(cameraId) && cameraId != "ALL")
            q = q.Where(e => e.CameraId == cameraId);

        if (!string.IsNullOrWhiteSpace(verdictFilter) && verdictFilter != "All")
        {
            if (verdictFilter == "Unreviewed")
                q = q.Where(e => e.Review == null);
            else
                q = q.Where(e => e.Review != null && e.Review.Verdict == verdictFilter);
        }

        return await q.OrderByDescending(e => e.CreatedAtUtc).Take(500).ToListAsync(ct);
    }

    public async Task SubmitReviewAsync(
        string eventId,
        string verdict,
        string note,
        string reviewedBy = "CounterStaff",
        int? userId = null,
        CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var existing = await db.EventReviews.FirstOrDefaultAsync(r => r.EventId == eventId, ct);
        if (existing is null)
        {
            db.EventReviews.Add(new EventReview
            {
                EventId = eventId,
                UserId = userId,
                ReviewedBy = reviewedBy,
                Verdict = verdict,
                Note = note ?? string.Empty,
                ReviewedAtUtc = DateTime.UtcNow,
            });
        }
        else
        {
            existing.UserId = userId;
            existing.ReviewedBy = reviewedBy;
            existing.Verdict = verdict;
            existing.Note = note ?? string.Empty;
            existing.ReviewedAtUtc = DateTime.UtcNow;
        }
        await db.SaveChangesAsync(ct);
    }

    public async Task<bool> VerifyOwnerPinAsync(string pin, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var admin = await db.Users.AsNoTracking()
            .Where(u => (u.Username == "admin" || u.Username == "owner") && u.IsActive)
            .OrderByDescending(u => u.Username == "admin")
            .FirstOrDefaultAsync(ct);
        return admin is not null && PinHasher.VerifyPin(pin, admin.PinHash);
    }

    public Task<bool> IsUsingDefaultPinAsync(CancellationToken ct = default)
        => VerifyOwnerPinAsync("1234", ct);

    public async Task DeleteReviewAsync(string eventId, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var existing = await db.EventReviews.FirstOrDefaultAsync(r => r.EventId == eventId, ct);
        if (existing is not null)
        {
            db.EventReviews.Remove(existing);
            await db.SaveChangesAsync(ct);
        }
    }

    public async Task UpdateOwnerPinAsync(string newPin, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        var admin = await db.Users
            .OrderByDescending(u => u.Username == "admin")
            .FirstOrDefaultAsync(u => u.Username == "admin" || u.Username == "owner", ct);
        if (admin is null)
        {
            db.Users.Add(new UserAccount
            {
                Username = "admin",
                DisplayName = "Store Admin",
                Role = "Admin",
                PinHash = PinHasher.HashPin(newPin),
                IsActive = true,
            });
        }
        else
        {
            admin.DisplayName = "Store Admin";
            admin.Role = "Admin";
            admin.PinHash = PinHasher.HashPin(newPin);
        }
        await db.SaveChangesAsync(ct);
    }

    public static string BuildCsvReport(IEnumerable<EventRecord> records)
    {
        var sb = new StringBuilder();
        sb.AppendLine("EventId,CreatedAtUtc,CameraId,CameraName,TrackId,Category,ConfidencePct,Verdict,ReviewedBy,Note,Justification,GridPath,ClipPath");
        foreach (var e in records)
        {
            static string Esc(string? s) => "\"" + (s ?? string.Empty).Replace("\"", "\"\"") + "\"";
            sb.AppendLine(string.Join(",",
                Esc(e.Id),
                Esc(e.CreatedAtUtc.ToString("O")),
                Esc(e.CameraId),
                Esc(e.CameraName),
                e.TrackId.ToString(),
                Esc(e.BehaviorCategory),
                e.ConfidencePct.ToString("F2"),
                Esc(e.Review?.Verdict ?? "Unreviewed"),
                Esc(e.Review?.ReviewedBy ?? string.Empty),
                Esc(e.Review?.Note ?? string.Empty),
                Esc(e.Justification),
                Esc(e.GridPath),
                Esc(e.ClipPath)
            ));
        }
        return sb.ToString();
    }
}
