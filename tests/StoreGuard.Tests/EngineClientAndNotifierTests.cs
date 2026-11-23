using System.Net;
using System.Net.Http;
using System.Text;
using Microsoft.EntityFrameworkCore;
using StoreGuard.Core.Models;
using StoreGuard.Infrastructure.Engine;
using StoreGuard.Infrastructure.Notifications;
using StoreGuard.Infrastructure.Persistence;
using StoreGuard.Infrastructure.Sdk;
using StoreGuard.Infrastructure.Security;
using Xunit;

namespace StoreGuard.Tests;

internal sealed class StubHttpHandler : HttpMessageHandler
{
    private readonly Func<HttpRequestMessage, HttpResponseMessage> _fn;
    private readonly object _gate = new();
    public List<HttpRequestMessage> Requests { get; } = new();
    public List<string> Bodies { get; } = new();

    public StubHttpHandler(Func<HttpRequestMessage, HttpResponseMessage> fn) => _fn = fn;

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        lock (_gate)
            Requests.Add(request);
        if (request.Content is not null)
        {
            var body = await request.Content.ReadAsStringAsync(cancellationToken);
            lock (_gate)
                Bodies.Add(body);
        }
        return _fn(request);
    }
}

public sealed class EngineClientAndNotifierTests : IDisposable
{
    private readonly string _tempDir;

    public EngineClientAndNotifierTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sg_cli_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void TheftEngineClient_DispatchesEnvelopes()
    {
        var client = new TheftEngineClient("http://127.0.0.1:8700");
        TheftAlertEvent? gotAlert = null;
        CameraHealthEvent? gotHealth = null;
        VlmErrorEvent? gotErr = null;

        client.OnTheftAlert += (_, a) => gotAlert = a;
        client.OnCameraHealth += (_, h) => gotHealth = h;
        client.OnVlmError += (_, e) => gotErr = e;

        client.DispatchEnvelope("""{"type":"ALERT","timestamp_utc":"2026-11-21T01:00:00Z","payload":{"event_id":"e1","camera_id":"cam_01","camera_name":"A1","track_id":3,"behavior_category":"high_risk_concealment_attempt","confidence_pct":82.0,"behaviors":["reach_and_tuck"],"nearby_bags":[],"justification":"Placed item in bag","grid_path":"g.jpg","clip_path":"c.mp4"}}""");
        client.DispatchEnvelope("""{"type":"CAMERA_HEALTH","timestamp_utc":"2026-11-21T01:00:00Z","payload":{"camera_id":"cam_01","state":"ONLINE","fps":8.2,"motion_active":true,"last_frame_age_ms":50,"error":""}}""");
        client.DispatchEnvelope("""{"type":"VLM_ERROR","timestamp_utc":"2026-11-21T01:00:00Z","payload":{"camera_id":"cam_02","track_id":1,"error":"HTTP 504","grid_path":"c.jpg"}}""");

        Assert.NotNull(gotAlert);
        Assert.Equal(82.0, gotAlert!.ConfidencePct);
        Assert.NotNull(gotHealth);
        Assert.Equal("ONLINE", gotHealth!.State);
        Assert.NotNull(gotErr);
        Assert.Equal("HTTP 504", gotErr!.Error);
    }

    [Fact]
    public async Task TheftEngineClient_RestRequests_WorkAgainstMockHandler()
    {
        var handler = new StubHttpHandler(req =>
        {
            if (req.RequestUri!.AbsolutePath == "/health")
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"status":"ready","device":"cpu","using_gpu":false,"active_streams":2,"vlm_queue_depth":0,"cameras":[]}""", Encoding.UTF8, "application/json"),
                };
            }
            if (req.RequestUri.AbsolutePath == "/cameras/cam_01/zones" && req.Method == HttpMethod.Put)
                return new HttpResponseMessage(HttpStatusCode.OK);
            if (req.RequestUri.AbsolutePath == "/cameras/cam_01/rerun" && req.Method == HttpMethod.Post)
                return new HttpResponseMessage(HttpStatusCode.OK);
            if (req.RequestUri.AbsolutePath == "/engine/device" && req.Method == HttpMethod.Put)
            {
                var raw = req.Content!.ReadAsStringAsync().Result;
                var reqDev = raw.Contains("cpu") ? "cpu" : "cuda:0";
                var respJson = $"{{\"requested_device\":\"{reqDev}\",\"resolved_device\":\"cpu\",\"using_gpu\":false,\"cuda_available\":false,\"fallback_note\":\"\"}}";
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(respJson, Encoding.UTF8, "application/json"),
                };
            }
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        await using var client = new TheftEngineClient("http://127.0.0.1:8700", handler);
        var health = await client.GetHealthAsync(CancellationToken.None);
        Assert.NotNull(health);
        Assert.Equal(2, health!.ActiveStreams);

        var ok = await client.UpdateCameraZonesAsync("cam_01", [new CameraZoneDto { Name = "shelf_1", Role = "shelf", RectNorm = [0.1, 0.1, 0.2, 0.4] }], CancellationToken.None);
        Assert.True(ok);

        Assert.True(await client.RerunCameraDetectionAsync("cam_01", CancellationToken.None));
        Assert.False(await client.RerunCameraDetectionAsync("cam_missing", CancellationToken.None));

        var devCpu = await client.SetDeviceAsync("cpu", CancellationToken.None);
        Assert.NotNull(devCpu);
        Assert.Equal("cpu", devCpu!.RequestedDevice);
        var devGpu = await client.SetDeviceAsync("cuda:0", CancellationToken.None);
        Assert.NotNull(devGpu);
        Assert.Equal("cuda:0", devGpu!.RequestedDevice);
    }

    [Fact]
    public async Task TheftEngineClient_GetSnapshotAsync_SendsOptionalSdDownscaleQueryParams()
    {
        var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new ByteArrayContent([0xFF, 0xD8, 0xFF, 0xD9]),
        });

        await using var client = new TheftEngineClient("http://127.0.0.1:8700", handler);
        var sd = await client.GetSnapshotAsync("cam_01", true, CancellationToken.None, maxWidth: 640, quality: 75);
        var hd = await client.GetSnapshotAsync("cam_01", false, CancellationToken.None);

        Assert.NotNull(sd);
        Assert.NotNull(hd);
        Assert.Equal("?overlays=true&max_width=640&quality=75", handler.Requests[0].RequestUri!.Query);
        Assert.Equal("?overlays=false", handler.Requests[1].RequestUri!.Query);
    }

    [Fact]
    public async Task TelegramNotifier_SendsPhotoImmediately_WhenEnabled()
    {
        var dbPath = Path.Combine(_tempDir, "tg.db");
        var repo = new EventRepository(new TestDbFactory(dbPath));
        await repo.InitializeAsync();
        var protector = new DpapiSecretProtector();

        var settings = await repo.GetSettingsAsync();
        settings.TelegramEnabled = true;
        settings.EncryptedTelegramBotToken = protector.Protect("123:ABC");
        settings.TelegramChatId = "999888";
        await repo.SaveSettingsAsync(settings);

        var gridPath = Path.Combine(_tempDir, "evt_grid.jpg");
        await File.WriteAllBytesAsync(gridPath, [0xFF, 0xD8, 0xFF, 0xD9]);

        var handler = new StubHttpHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        var notifier = new TelegramNotifier(repo, protector, new HttpClient(handler));
        var (ok, _) = await notifier.SendTheftAlertAsync(new TheftAlertEvent
        {
            EventId = "evt_tg",
            CameraId = "cam_01",
            CameraName = "Cosmetics",
            TrackId = 2,
            ConfidencePct = 91.0,
            Behaviors = ["reach_and_tuck"],
            Justification = "Item placed in bag",
            GridPath = gridPath,
        }, CancellationToken.None);

        Assert.True(ok);
        Assert.Single(handler.Requests);
        Assert.EndsWith("/sendPhoto", handler.Requests[0].RequestUri!.AbsolutePath);
    }

    [Fact]
    public async Task PythonProcessHost_StartsRealAiEngine_ConnectsWebSocket_AndShutsDownCleanly()
    {
        var cur = new DirectoryInfo(AppContext.BaseDirectory);
        string? rootDir = null;
        while (cur is not null)
        {
            if (Directory.Exists(Path.Combine(cur.FullName, "ai_engine")) && File.Exists(Path.Combine(cur.FullName, "StoreGuard.sln")))
            {
                rootDir = cur.FullName;
                break;
            }
            cur = cur.Parent;
        }
        Assert.NotNull(rootDir);

        var cfgPath = Path.Combine(_tempDir, "e2e_cameras.json");
        await File.WriteAllTextAsync(cfgPath, """
{
  "engine": { "device": "cpu", "target_fps": 4, "vlm": { "provider": "dry_run" } },
  "cameras": []
}
""");

        const int testPort = 8711;
        await using var host = new PythonProcessHost(rootDir!, cfgPath, port: testPort, logDir: Path.Combine(_tempDir, "logs"));
        await host.StartAndMonitorAsync(CancellationToken.None);
        bool ready = await host.WaitForHealthyAsync(TimeSpan.FromSeconds(25), CancellationToken.None);
        Assert.True(ready, "Real Python ai_engine subprocess failed to report /health within 25s.");

        await using var client = new TheftEngineClient($"http://127.0.0.1:{testPort}");
        var health = await client.GetHealthAsync(CancellationToken.None);
        Assert.NotNull(health);
        Assert.Equal("ready", health!.Status);
        Assert.Equal("apex_vision", health.Runtime);

        var sdk = new ApexSolution(client);
        var solInfo = await sdk.RefreshAsync(CancellationToken.None);
        Assert.NotNull(solInfo);
        Assert.Equal("apex_vision", solInfo!.Runtime);

        var connectedTcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        client.OnConnectionChanged += (_, isUp) => { if (isUp) connectedTcs.TrySetResult(true); };
        await client.StartListeningAsync(CancellationToken.None);
        Assert.True(await Task.WhenAny(connectedTcs.Task, Task.Delay(5000)) == connectedTcs.Task);
        Assert.True(client.IsConnected);

        // DisposeAsync sends /shutdown and ensures clean exit without leaving orphaned processes
        await client.DisposeAsync();
        await host.DisposeAsync();
        await Task.Delay(300);
        Assert.Null(await client.GetHealthAsync(CancellationToken.None));
    }

    [Fact]
    public async Task ApexSolutionSdk_BindsProcedures_MutatesParams_AndRunsOnce()
    {
        var seenPaths = new List<string>();
        var handler = new StubHttpHandler(req =>
        {
            var path = req.RequestUri!.AbsolutePath;
            seenPaths.Add($"{req.Method.Method} {path}");
            if (path == "/sdk/solution")
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""
                        {"name":"StoreGuard","solution_path":"storeguard.apex","runtime":"apex_vision","procedures":[{"name":"cam_01","enabled":true,"execution_count":4,"avg_time_ms":2.5,"connections_count":18,"nodes":[{"instance_id":"node_motion","name":"03_Motion_Gate","tool_name":"Motion Gate","enabled":true,"parameters":{"diff_thresh":16}}]}]}
                        """, Encoding.UTF8, "application/json"),
                };
            }
            if (path == "/sdk/procedures/cam_01/params/node_motion/diff_thresh" && req.Method == HttpMethod.Put)
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""
                        {"name":"cam_01","enabled":true,"execution_count":4,"avg_time_ms":2.5,"connections_count":18,"nodes":[{"instance_id":"node_motion","name":"03_Motion_Gate","tool_name":"Motion Gate","enabled":true,"parameters":{"diff_thresh":24}}]}
                        """, Encoding.UTF8, "application/json"),
                };
            }
            if (path == "/sdk/procedures/cam_01/run" && req.Method == HttpMethod.Post)
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""
                        {"procedure":"cam_01","executed":true,"elapsed_ms":0.42,"triggered":false,"armed":false,"node_statuses":{"node_motion":"OK","node_pose":"BYPASS"}}
                        """, Encoding.UTF8, "application/json"),
                };
            }
            if (path == "/sdk/solution/xml")
            {
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent("""{"name":"StoreGuard","solution_path":"storeguard.apex","xml":"<ApexSolution name=\"StoreGuard\" />"}""", Encoding.UTF8, "application/json"),
                };
            }
            return new HttpResponseMessage(HttpStatusCode.NotFound);
        });

        await using var client = new TheftEngineClient("http://127.0.0.1:8799", handler);
        var solution = new ApexSolution(client);
        var info = await solution.RefreshAsync();
        Assert.NotNull(info);
        var proc = solution["cam_01"];
        Assert.Equal("cam_01", proc.Name);

        var updated = await proc.SetParamAsync("node_motion", "diff_thresh", 24);
        Assert.NotNull(updated);
        Assert.Equal(24, int.Parse(updated!.Nodes[0].Parameters["diff_thresh"]!.ToString()!));

        var run = await proc.RunAsync();
        Assert.NotNull(run);
        Assert.True(run!.Executed);
        Assert.Equal("BYPASS", run.NodeStatuses["node_pose"]);

        var xml = await solution.ExportXmlAsync();
        Assert.Contains("<ApexSolution", xml);
        Assert.Contains("PUT /sdk/procedures/cam_01/params/node_motion/diff_thresh", seenPaths);
    }

    public void Dispose()
    {
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(_tempDir, true); } catch { }
    }
}
