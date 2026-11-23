using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using StoreGuard.Core.Interfaces;
using StoreGuard.Core.Models;

namespace StoreGuard.Infrastructure.Sdk;

/// <summary>
/// Hikrobot VisionMaster-style VmProcedure handle bound to one Apex Vision camera flow.
/// </summary>
public sealed class ApexProcedure
{
    private readonly IApexSolutionClient _client;

    internal ApexProcedure(IApexSolutionClient client, ApexProcedureSummary summary)
    {
        _client = client;
        Summary = summary;
    }

    public string Name => Summary.Name;
    public ApexProcedureSummary Summary { get; private set; }
    public bool Enabled => Summary.Enabled;
    public IReadOnlyList<ApexProcedureNodeInfo> Nodes => Summary.Nodes;

    public Task<ApexProcedureRunResult?> RunAsync(CancellationToken ct = default) =>
        _client.RunProcedureOnceAsync(Name, ct);

    public async Task<ApexProcedureSummary?> RefreshAsync(CancellationToken ct = default)
    {
        var updated = await _client.GetProcedureAsync(Name, ct).ConfigureAwait(false);
        if (updated is not null)
            Summary = updated;
        return updated;
    }

    public async Task<ApexProcedureSummary?> SetParamAsync(
        string nodeId,
        string paramName,
        object? value,
        CancellationToken ct = default)
    {
        var updated = await _client.SetProcedureParamAsync(Name, nodeId, paramName, value, ct).ConfigureAwait(false);
        if (updated is not null)
            Summary = updated;
        return updated;
    }
}

/// <summary>
/// Hikrobot VisionMaster-style VmSolution container managing multi-camera <ApexFlow> procedures.
/// Usage: var proc = solution["cam_01"]; await proc.SetParamAsync("node_motion", "diff_thresh", 20);
/// </summary>
public sealed class ApexSolution
{
    private readonly IApexSolutionClient _client;
    private readonly Dictionary<string, ApexProcedure> _procedures = new(StringComparer.OrdinalIgnoreCase);

    public ApexSolution(IApexSolutionClient client)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
    }

    public string Name { get; private set; } = "StoreGuard";
    public string SolutionPath { get; private set; } = string.Empty;
    public string Runtime { get; private set; } = "apex_vision";

    public IReadOnlyCollection<string> ProcedureNames => _procedures.Keys.ToList();

    public ApexProcedure this[string procedureName]
    {
        get
        {
            if (_procedures.TryGetValue(procedureName, out var proc))
                return proc;
            throw new KeyNotFoundException($"Apex procedure '{procedureName}' is not loaded in solution '{Name}'.");
        }
    }

    public bool TryGetProcedure(string procedureName, out ApexProcedure? procedure) =>
        _procedures.TryGetValue(procedureName, out procedure);

    public async Task<ApexSolutionInfo?> RefreshAsync(CancellationToken ct = default)
    {
        var info = await _client.GetSolutionAsync(ct).ConfigureAwait(false);
        ApplyInfo(info);
        return info;
    }

    public async Task<ApexSolutionInfo?> LoadAsync(string solutionPath, CancellationToken ct = default)
    {
        var info = await _client.LoadSolutionAsync(path: solutionPath, xml: null, ct).ConfigureAwait(false);
        ApplyInfo(info);
        return info;
    }

    public async Task<ApexSolutionInfo?> LoadXmlAsync(string xmlContent, CancellationToken ct = default)
    {
        var info = await _client.LoadSolutionAsync(path: null, xml: xmlContent, ct).ConfigureAwait(false);
        ApplyInfo(info);
        return info;
    }

    public Task<ApexSolutionSaveResponse?> SaveAsync(string? path = null, CancellationToken ct = default) =>
        _client.SaveSolutionAsync(path, ct);

    public async Task<string> ExportXmlAsync(CancellationToken ct = default)
    {
        var res = await _client.GetSolutionXmlAsync(ct).ConfigureAwait(false);
        return res?.Xml ?? string.Empty;
    }

    private void ApplyInfo(ApexSolutionInfo? info)
    {
        if (info is null)
            return;
        Name = info.Name;
        SolutionPath = info.SolutionPath;
        Runtime = info.Runtime;
        _procedures.Clear();
        foreach (var summary in info.Procedures)
            _procedures[summary.Name] = new ApexProcedure(_client, summary);
    }
}
