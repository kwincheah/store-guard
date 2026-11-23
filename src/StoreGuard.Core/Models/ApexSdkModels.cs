using System.Text.Json.Serialization;

namespace StoreGuard.Core.Models;

public sealed class ApexProcedureNodeInfo
{
    [JsonPropertyName("instance_id")]
    public string InstanceId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("tool_name")]
    public string ToolName { get; set; } = string.Empty;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; } = true;

    [JsonPropertyName("parameters")]
    public Dictionary<string, object?> Parameters { get; set; } = new();
}

public sealed class ApexProcedureSummary
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; } = true;

    [JsonPropertyName("execution_count")]
    public int ExecutionCount { get; set; }

    [JsonPropertyName("avg_time_ms")]
    public double AvgTimeMs { get; set; }

    [JsonPropertyName("nodes")]
    public List<ApexProcedureNodeInfo> Nodes { get; set; } = new();

    [JsonPropertyName("connections_count")]
    public int ConnectionsCount { get; set; }
}

public sealed class ApexSolutionInfo
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("solution_path")]
    public string SolutionPath { get; set; } = string.Empty;

    [JsonPropertyName("runtime")]
    public string Runtime { get; set; } = "apex_vision";

    [JsonPropertyName("procedures")]
    public List<ApexProcedureSummary> Procedures { get; set; } = new();
}

public sealed class ApexSolutionXmlResponse
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("solution_path")]
    public string SolutionPath { get; set; } = string.Empty;

    [JsonPropertyName("xml")]
    public string Xml { get; set; } = string.Empty;
}

public sealed class ApexSolutionLoadRequest
{
    [JsonPropertyName("path")]
    public string? Path { get; set; }

    [JsonPropertyName("xml")]
    public string? Xml { get; set; }
}

public sealed class ApexSolutionSaveResponse
{
    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("solution_path")]
    public string SolutionPath { get; set; } = string.Empty;
}

public sealed class ApexProcedureRunResult
{
    [JsonPropertyName("procedure")]
    public string Procedure { get; set; } = string.Empty;

    [JsonPropertyName("executed")]
    public bool Executed { get; set; }

    [JsonPropertyName("elapsed_ms")]
    public double ElapsedMs { get; set; }

    [JsonPropertyName("node_statuses")]
    public Dictionary<string, string> NodeStatuses { get; set; } = new();

    [JsonPropertyName("triggered")]
    public bool Triggered { get; set; }

    [JsonPropertyName("armed")]
    public bool Armed { get; set; }

    [JsonPropertyName("error")]
    public string? Error { get; set; }
}

public sealed class ApexParamUpdateRequest
{
    [JsonPropertyName("value")]
    public object? Value { get; set; }
}
